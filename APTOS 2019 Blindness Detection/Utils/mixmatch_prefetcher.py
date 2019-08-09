import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class data_prefetcher():
    def __init__(self, loader, mode='train'):
        assert mode in ['train', 'valid', 'test', 'TransformTwice']
        self.loader_ = iter(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mode = mode
        self.preload()

    def preload(self):
        if self.mode in ['train', 'valid']:
            try:
                self.data = next(self.loader)
                self.next_input, self.next_target = self.data['image'], self.data['label'].view(-1, 1)
            except StopIteration:
                # self.next_input = None
                # self.next_target
                # return
                if self.mode == 'valid':
                    self.next_input = None
                    self.next_target = None
                    return
                else:
                    self.data = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(device, dtype=torch.float, non_blocking=True)
                self.next_target = self.next_target.to(device, dtype=torch.float, non_blocking=True)
        elif self.mode == 'test':
            try:
                self.image = next(self.loader)
                self.next_input = self.image
            except StopIteration:
                self.next_input = None
                return
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(device, dtype=torch.float, non_blocking=True)
        else:
            try:
                self.next_input, self.next_input_1 = next(self.loader)
            except:
                # self.next_input, self.next_input_1 = next(self.loader_)
                self.next_input, self.next_input_1 = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.to(device, dtype=torch.float, non_blocking=True)
                self.next_input_1 = self.next_input_1.to(device, dtype=torch.float, non_blocking=True)

    def next(self):
        if self.mode in ['train', 'valid']:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
            self.preload()
            return input, target
        elif self.mode == 'test':
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            self.preload()
            return input
        else:
            torch.cuda.current_stream().wait_stream(self.stream)
            input_0 = self.next_input
            input_1 = self.next_input_1
            if input_0 is not None:
                input_0.record_stream(torch.cuda.current_stream())
            if input_1 is not None:
                input_1.record_stream(torch.cuda.current_stream())
            self.preload()
            return input_0, input_1
