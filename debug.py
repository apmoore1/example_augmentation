from pathlib import Path
from allennlp.common import Params
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.training.optimizers import Optimizer
from allennlp.models import Model, load_archive
import bella_allen_nlp


model_fp = str(Path('.', 'model_configs', 'standard', 'ds_elmo_t_fine_tune_laptop.json').resolve())
params = Params.from_file(model_fp)
data_fp = str(Path('data', 'splits', 'Laptop Test').resolve())
reader = DatasetReader.from_params(params['dataset_reader'])
instances = list(reader.read(data_fp))

if 'vocabulary' in params:
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
else:
    vocab = Vocabulary.from_instances(instances)
import re
model = Model.from_params(vocab=vocab, params=params['model'])
names = [name for name, param in model.named_parameters()]
optim = Optimizer.from_params(model.named_parameters(),params['trainer']['optimizer'])
values = {}
for i, param_group in enumerate(reversed(optim.param_groups)):
    values[i] = param_group['params']
        
encoder = set()
project = set()
embedder_5 = set()
embedder_4 = set()
embedder_3 = set()
embedder_2 = set()
embedder_1 = set()
embedder_0 = set()
embedder_char = set()
embedding_scalar_mix = set()
other = set()
for name in names:
    if re.search('text_encoder.*', name):
        encoder.add(name)
    elif re.search('target_encoder.*', name):
        encoder.add(name)
    elif re.search('label_projection.*', name):
        project.add(name)
    elif re.search('.*scalar_mix.*', name):
        embedding_scalar_mix.add(name)
    elif re.search('text_field_embedder.*contextualizer.*layers.5.*', name):
        embedder_5.add(name)
    elif re.search('text_field_embedder.*contextualizer.*layers.4.*', name):
        embedder_4.add(name)
    elif re.search('text_field_embedder.*contextualizer.*layers.3.*', name):
        embedder_3.add(name)
    elif re.search('text_field_embedder.*contextualizer.*layers.2.*', name):
        embedder_2.add(name)
    elif re.search('text_field_embedder.*contextualizer.*layers.1.*', name):
        embedder_1.add(name)
    elif re.search('text_field_embedder.*contextualizer.*layers.0.*', name):
        embedder_0.add(name)
    elif re.search('text_field_embedder.*token_characters.*', name):
        embedder_char.add(name)
    elif re.search(".*backward_transformer.norm.beta", name):
        other.add(name)
    elif re.search(".*forward_transformer.norm.beta", name):
        other.add(name)
    elif re.search(".*forward_transformer.norm.gamma", name):
        other.add(name)
    elif re.search(".*backward_transformer.norm.gamma", name):
        other.add(name)
    else:
        raise ValueError('it')
import pdb
pdb.set_trace()