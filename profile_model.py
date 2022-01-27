import argparse
from pathlib import Path
import itertools
from timeit import default_timer as timer
from datetime import timedelta
from torch.profiler import profile, record_function, ProfilerActivity

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from InferenceInterfaces.InferenceArchitectures.InferenceTransformerTTS import Transformer
from Preprocessing.TextFrontend import TextFrontend


attention_methods = ["multihead_softmax_att", "performer", "reformer", "linformer"]  # TODO: add your attention here


def collate_and_pad(batch):
        texts = [torch.LongTensor(text).squeeze(0) for text in batch]
        return pad_sequence(texts, batch_first=True)


class CustomDataset(Dataset):
    # A pytorch dataset class for holding data for a text classification task.
    def __init__(self, filename):
        '''
        Takes as input the name of a file containing sentences with a classification label (comma separated) in each line.
        Stores the text data in a member variable X and labels in y
        '''
        self.X = []
        # Opening the file and storing its contents in a list
        with open(filename) as f:
            for line in f:
                self.X.append(line)

        self.text2phone = TextFrontend(language="en", use_word_boundaries=False,
                                       use_explicit_eos=False, inference=True)

    def preprocess(self, text):
        ### Do something with text here
        text_pp = text.lower().strip()
        ###

        # transform the string into phones
        return self.text2phone.string_to_tensor(text).squeeze(0).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        '''
        Returns the text present at the specified index of the lists.
        '''

        return self.preprocess(self.X[index])


def trace_handler(p):
    print('in trace handler')
    output = p.key_averages().table(sort_by='cpu_memory_usage', row_limit=10)
    print("output:", output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
 

def run_profiling(model_name, attention, device, data):
    if model_name.endswith(".pt"):
        model_path = model_name
    else:
        model_path = Path("Models", model_name, "best.pt")

    # TODO: you can control what you want to test by adapting this dict
    # device: ['gpu', 'cpu'], mode: ['training', 'inference'], utterances: ['single', 'batch']
    options = {
        'device': ['gpu'],  # delete 'gpu' if you want to run it only on CPU
        'mode': ['inference'],
        'utterances': ['single'],
    }

    # we will test all possible combinations of the options defined above
    option_combinations = list(itertools.product(options['device'], options['mode'], options['utterances']))
    custom_checkpoints = []

    for config in option_combinations:
        prof_device, prof_mode, prof_utterances = config

        if prof_device == 'gpu':
            current_device = torch.device(f'cuda:{device}')
        else:
            current_device = torch.device('cpu')

        checkpoint_name = f"on_{prof_device}-{prof_mode}_mode-{prof_utterances}_utterances"
        print(f'run checkpoint: {checkpoint_name}')
        custom_checkpoints.append(checkpoint_name)


        # TODO: change the "decoder_self_att_type" parameter to whatever you used to specify the attention method
        model = Transformer(path_to_weights=model_path,
                            idim=166, odim=80, spk_embed_dim=None, reduction_factor=1).to(current_device)

        if prof_mode == 'training':
            model.train()
            execute_profiling(model, data, prof_utterances, current_device)
        else:
            model.eval()
            with torch.no_grad():
                execute_profiling(model, data, prof_utterances, current_device)
    print('profiling successfully terminated')



def execute_profiling(model, data, prof_utterances, device):
    """
    run the pytorch profiler with a schedule to ensure that the extra set-up time doesn't impact the overall stats
    
    :param model: the trained model you want to profile 
    :param data: a dataloader
    :param prof_utterances: a string with the value 'single' or 'batch'
    :param device: the device the model is set to run on e.g. cpu or cuda:0 etc.
    :return: none, writes a log file which can be loaded via tensorboard
    """
    #TODO change the log address if wanted
    with profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=2), 
             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
             profile_memory=True, record_shapes=True) as prof:
        for text_batch in data:
            if prof_utterances == 'single':
                for text in text_batch:
                    model(text.to(device))
                    prof.step()
            else:
                print("text_batch:", text_batch)
                model(text_batch.to(device))
                prof.step()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='IMS Speech Synthesis Toolkit - Call to Train')

    parser.add_argument('model_name',
                        type=str,
                        help="Name of the trained Transformer model. If the name ends with '.pt', it is assumed to "
                             "give the full path to the model, otherwise the name of a directory inside of 'Models' "
                             "in which there is a trained model 'best.pt'.")

    parser.add_argument('attention',
                        choices=attention_methods,
                        help="Self attention method for the decoder.")

    parser.add_argument('--test_sentences',
                        type=str,
                        help="Path to test sentences.",
                        default="test_sentences/small_test.txt")

    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    args = parser.parse_args()

    dataset = CustomDataset(args.test_sentences)
    dataloader = DataLoader(batch_size=2,
                            dataset=dataset,
                            num_workers=8,
                            collate_fn=collate_and_pad,
                            pin_memory=True
                            )

    run_profiling(model_name=args.model_name, attention=args.attention, device=args.gpu_id, data=dataloader)
