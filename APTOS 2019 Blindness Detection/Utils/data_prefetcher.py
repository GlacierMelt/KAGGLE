import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class data_prefetcher():
    def __init__(self, loader, mode='train'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mode = mode
        self.preload()

    def preload(self):
        if self.mode == 'train':
            try:
                self.data = next(self.loader)
                self.next_input, self.next_target = self.data['image'], self.data['label'].view(-1, 1)
            except StopIteration:
                self.next_input = None
                self.next_target = None
                return
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(device, dtype=torch.float, non_blocking=True)
                self.next_target = self.next_target.to(device, dtype=torch.float, non_blocking=True)
        else:
            try:
                self.data = next(self.loader)
                self.next_input = self.data['image']
            except StopIteration:
                self.next_input = None
                return
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(device, dtype=torch.float, non_blocking=True)

    def next(self):
        if self.mode == 'train':
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
            self.preload()
            return input, target
        else:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            self.preload()
            return input
