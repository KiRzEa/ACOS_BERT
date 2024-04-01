import torch
from torchcrf import CRF
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import BertModel


class BertForTABSAJoint_CRF(nn.Module):

	def __init__(self, model_name, config, num_labels, num_ner_labels):
		super(BertForTABSAJoint_CRF, self).__init__()
		self.bert = BertModel.from_pretrained(model_name)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		# CRF
		self.CRF_model = CRF(num_ner_labels, batch_first=True)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, nn.LayerNorm):
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
				module.bias.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, attention_mask, ner_mask, acs_labels=None, ner_labels=None, inference=False):
		outputs = self.bert(input_ids, attention_mask)
		sequence_output = outputs.last_hidden_state
		pooled_output = outputs.pooler_output
		# get the last hidden layer
		
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		ner_logits = self.ner_hidden2tag(sequence_output)
		
		# the CRF layer of NER labels

		ner_predict = self.CRF_model.decode(ner_logits, ner_mask.type(torch.ByteTensor).cuda())
		
		# the classifier of category & polarity
		if not inference:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, acs_labels)
			ner_loss_list = self.CRF_model(ner_logits, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
			ner_loss = torch.mean(-ner_loss_list)
			
			return loss, ner_loss, logits, ner_predict
		else:
			return logits, ner_predict