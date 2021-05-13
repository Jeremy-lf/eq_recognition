'''
Python 3.6 
Pytorch >= 0.4
'''
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict

GroupNorm_num = 32
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1_G', nn.GroupNorm(GroupNorm_num,num_input_features)),
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2_G', nn.GroupNorm(GroupNorm_num,bn_size * growth_rate)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.add_module('norm_G', nn.GroupNorm(GroupNorm_num,num_input_features))
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0_m', nn.Conv2d(2, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            # ('norm0_G', nn.GroupNorm(GroupNorm_num,num_init_features)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5_G', nn.GroupNorm(GroupNorm_num,num_features))
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.1
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        import numpy
        # orig_x = x.numpy()
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=2)
        # out = self.classifier(out)
        return out


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6,12,24),
                     **kwargs)
    return model

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.embedding = nn.Embedding(self.output_size, 256)
        #self.gru = nn.GRUCell(684, 256)
        self.gru = nn.GRUCell(1024, self.hidden_size)
        self.gru1 = nn.GRUCell(256, self.hidden_size)
        self.out = nn.Linear(128, self.output_size)
        self.hidden = nn.Linear(self.hidden_size, 256)
        self.emb = nn.Linear(256, 128)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_et = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.conv_tan = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.hidden2 = nn.Linear(self.hidden_size, 128)
        self.emb2 = nn.Linear(256, 128)
        self.ua = nn.Linear(1024, 256)
        self.uf = nn.Linear(1, 256)
        self.v = nn.Linear(256, 1)
        self.wc = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input_a, hidden, encoder_outputs,bb,attention_sum,decoder_attention,dense_input,batch_size,et_mask):

        et_mask_4 = et_mask.unsqueeze(1)
        # embedding the word from 1 to 256(total 112 words)
        embedded = self.embedding(input_a).view(batch_size,256)
        embedded = self.dropout(embedded)
        hidden = hidden.view(batch_size,self.hidden_size)

        st = self.gru1(embedded,hidden)
        hidden1 = self.hidden(st)
        hidden1 = hidden1.view(batch_size,1,1,256)

        # encoder_outputs from (batch,1024,height,width) => (batch,height,width,1024)
        encoder_outputs_trans = torch.transpose(encoder_outputs,1,2)
        encoder_outputs_trans = torch.transpose(encoder_outputs_trans,2,3)

        # encoder_outputs_trans (batch,height,width,1024) attention_sum_trans (batch,height,width,1) hidden1 (batch,1,1,256)
        decoder_attention = self.conv1(decoder_attention)
        attention_sum = attention_sum + decoder_attention
        attention_sum_trans = torch.transpose(attention_sum,1,2)
        attention_sum_trans = torch.transpose(attention_sum_trans,2,3)

        # encoder_outputs1 (batch,height,width,256) attention_sum1 (batch,height,width,256)
        encoder_outputs1 = self.ua(encoder_outputs_trans)
        attention_sum1 = self.uf(attention_sum_trans)

        et = hidden1 + encoder_outputs1 + attention_sum1
        et_trans = torch.transpose(et,2,3)
        et_trans = torch.transpose(et_trans,1,2)
        et_trans = self.conv_tan(et_trans)
        et_trans = et_trans*et_mask_4
        et_trans = self.bn1(et_trans)
        et_trans = torch.tanh(et_trans)
        et_trans = torch.transpose(et_trans,1,2)
        et_trans = torch.transpose(et_trans,2,3)

        et = self.v(et_trans) #4,9,34,1
        et = et.squeeze(3)

        # et_div_all is attention alpha
        et_div_all = et_mask.new(batch_size,1,dense_input,bb).zero_()
        # et_div_all = torch.zeros(batch_size,1,dense_input,bb).to(et_mask.get_device())
        et_div_all = et_div_all

        et_exp = torch.exp(et)
        et_exp = et_exp*et_mask
        et_sum = torch.sum(et_exp,dim=1)
        et_sum = torch.sum(et_sum,dim=1)
        for i in range(batch_size):
            et_div = et_exp[i]/(et_sum[i]+1e-8)
            et_div = et_div.unsqueeze(0)
            et_div_all[i] = et_div

        # ct is context vector (batch,128)
        ct = et_div_all*encoder_outputs
        ct = ct.sum(dim=2)
        ct = ct.sum(dim=2)

        # the next hidden after gru
        # batch,hidden_size
        hidden_next_a = self.gru(ct,st)
        hidden_next = hidden_next_a.view(batch_size, 1, self.hidden_size)

        # compute the output (batch,128)
        hidden2 = self.hidden2(hidden_next_a)
        embedded2 = self.emb2(embedded)
        ct2 = self.wc(ct)

        #output
        output = F.log_softmax(self.out(self.dropout(hidden2+embedded2+ct2)), dim=1)
        output = output.unsqueeze(1)

        return output, hidden_next, et_div_all, attention_sum

    def initHidden(self,batch_size):
        result = Variable(torch.randn(batch_size, 1, self.hidden_size))
        return result

class EquationRecognition:
    def __init__(self, gpu=False, device_id=0):
        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % device_id if device_id >= 0 else "cpu")
        else:
            self.device = torch.device("cpu")

        self.hidden_size = 256
        self.maxlen = 100
        self.model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_200117.pth')

        self.encoder = densenet121()
        # self.encoder = torch.nn.DataParallel(self.encoder)
        self.attn_decoder = AttnDecoderRNN(self.hidden_size, 330, dropout_p=0.5)

        stats = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        stats_en = self.encoder.state_dict()
        self.attn_decoder.load_state_dict(stats['decoder'])
        self.encoder.load_state_dict(stats['encoder'])

        self.worddicts_r = [None] * (len(stats['dictionary']) + 1)
        for kk, vv in stats['dictionary'].items():
            self.worddicts_r[vv] = kk

        self.encoder.to(self.device)
        self.encoder.eval()

        self.attn_decoder.to(self.device)
        self.attn_decoder.eval()
        
    def proc(self, gray_img):
        h_mask_t = []
        w_mask_t = []
        batch_size_t = 1

        gray = torch.from_numpy(np.array(gray_img)).type(torch.FloatTensor)
        gray = gray / 255.0
        gray = gray.unsqueeze(0)
        gray = gray.unsqueeze(0)

        x_t = Variable(gray)
        x_mask = torch.ones(x_t.size()[0], x_t.size()[1], x_t.size()[2], x_t.size()[3])
        x_t = torch.cat((x_t, x_mask), dim=1)

        img_size_h = x_t.size()[2]
        img_size_w = x_t.size()[3]

        if img_size_h > 171 and img_size_w > 672:
            padding_h = int(img_size_h / 171) * 2 + 1
            padding_w = int(img_size_w / 672) * 2 + 1
        else:
            padding_h = int(171 / img_size_h) * 4 + 1
            padding_w = int(672 / img_size_w) * 4 + 1
        m = torch.nn.ZeroPad2d((0, padding_w, 0, padding_h))
        x_t_1 = m(x_t)

        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]
        h_mask_t.append(int(x_real_high))
        w_mask_t.append(int(x_real_width))
        x_real = x_t[0][0].view(x_real_high, x_real_width)
        output_highfeature_t = self.encoder(x_t_1.to(self.device))

        x_mean_t = torch.mean(output_highfeature_t)
        x_mean_t = float(x_mean_t)
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[3]
        dense_input = output_area_t1[2]

        decoder_input_t = torch.LongTensor([111] * batch_size_t)
        decoder_input_t = decoder_input_t

        decoder_hidden_t = torch.randn(batch_size_t, 1, self.hidden_size)
        # nn.init.xavier_uniform_(decoder_hidden_t)
        decoder_hidden_t = decoder_hidden_t * x_mean_t
        decoder_hidden_t = torch.tanh(decoder_hidden_t)

        prediction = torch.zeros(batch_size_t, self.maxlen)
        # label = torch.zeros(batch_size_t,maxlen)

        decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t)
        attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t)
        decoder_attention_t_cat = []

        # batch_gpu must be an int object
        # batch_gpu = int(batch_size/len(gpu))
        et_mask = torch.zeros(batch_size_t, dense_input, output_area_t)
        for i in range(batch_size_t):
            et_mask[i][:h_mask_t[i],:w_mask_t[i]] = 1

        for i in range(self.maxlen):
            decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = self.attn_decoder(decoder_input_t.to(self.device),
                                                                                                   decoder_hidden_t.to(self.device),
                                                                                                   output_highfeature_t.to(self.device),
                                                                                                   output_area_t,
                                                                                                   attention_sum_t.to(self.device),
                                                                                                   decoder_attention_t.to(self.device),
                                                                                                   dense_input,
                                                                                                   batch_size_t,
                                                                                                   et_mask.to(self.device))

            decoder_attention_t_cat.append(decoder_attention_t[0].data.cpu().numpy())
            topv, topi = torch.max(decoder_output, 2)
            if torch.sum(topi) == 0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(batch_size_t)

            # prediction
            prediction[:, i] = decoder_input_t

        # k = np.array(decoder_attention_t_cat)
        # x_real = np.array(x_real.cpu().data)

        prediction = prediction[0]

        prediction_real = []
        for ir in range(len(prediction)):
            if int(prediction[ir]) == 0:
                break
            prediction_real.append(self.worddicts_r[int(prediction[ir])])
        # prediction_real.append('<eol>')

        prediction_real_show = np.array(prediction_real)
        pred = prediction_real_show.tolist()
        # real = ' '.join(value)

        while 'x' in pred and pred[pred.index('x') + 1].isdigit():
            pred[pred.index('x')] = '\\times'
        while 'x' in pred and pred[pred.index('x') + 1] == '\\underline':
            pred[pred.index('x')] = '\\times'
        while 'x' in pred and pred[pred.index('x') + 1] == '\left':
            pred[pred.index('x')] = '\\times'

        return ''.join(pred)


if __name__ == '__main__':

    encoder = densenet121()
    attn_decoder = AttnDecoderRNN(256, 330, dropout_p=0.5)
    # torch.onnx.export(encoder,torch.randn(1,2,224,224),"encoder.onnx",verbose=True)
    # torch.onnx.export(attn_decoder, torch.randn(1, 2, 224, 224), "attn_decoder.onnx", verbose=True)
