import torch

from .base import MFRecsysModel


class NMFRecsysModel(MFRecsysModel):
    def __init__(self, **kwargs):
        super(NMFRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.dropout = kwargs['dropout']

        self.embed_user_GMF = torch.nn.Embedding(kwargs['num_users'], kwargs['factor_num'])
        self.embed_item_GMF = torch.nn.Embedding(kwargs['num_items'], kwargs['factor_num'])
        self.embed_user_MLP = torch.nn.Embedding(kwargs['num_users'], kwargs['factor_num'] * (2 ** (kwargs['num_layers'] - 1)))
        self.embed_item_MLP = torch.nn.Embedding(kwargs['num_items'], kwargs['factor_num'] * (2 ** (kwargs['num_layers'] - 1)))

        MLP_modules = []
        for i in range(kwargs['num_layers']):
            input_size = kwargs['factor_num'] * (2 ** (kwargs['num_layers'] - i))
            MLP_modules.append(torch.nn.Dropout(p=self.dropout))
            MLP_modules.append(torch.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(torch.nn.ReLU())
        self.MLP_layers = torch.nn.Sequential(*MLP_modules)

        predict_size = kwargs['factor_num'] * 2
        self.predict_layer = torch.nn.Linear(predict_size, 1)

    def reset_parameters(self):
        torch.nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        torch.nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        torch.nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        embed_user_GMF = self.embed_user_GMF(user_indices)
        embed_item_GMF = self.embed_item_GMF(item_indices)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user_indices)
        embed_item_MLP = self.embed_item_MLP(item_indices)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
