import torch
import torch.distributed as dist


class LogitsQueue:
    def __init__(self, num_classes, queue_size=100):
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.queue_logits = torch.zeros(self.num_classes, self.queue_size)
        self.queue_ptrs = [0 for _ in range(self.num_classes)]
        self.counter = torch.zeros(self.num_classes)

    def enqueue_dequeue(self, cls_scores, gt_labels):
        for gt_label in gt_labels:
            self.counter[gt_label] += 1

        gt_labels_uniq = torch.unique(gt_labels)
        for label in gt_labels_uniq:
            if label == -1:
                continue
            cls_score = cls_scores[gt_labels == label][: self.queue_size]
            queue_ptr = self.queue_ptrs[label]
            score_size = len(cls_score)
            score_ptr = 0
            if queue_ptr + score_size > self.queue_size:
                score_ptr = self.queue_size - queue_ptr
                self.queue_logits[label, queue_ptr:] = cls_score[:score_ptr]
                score_size = score_size - score_ptr
                queue_ptr = 0
            self.queue_logits[label, queue_ptr : queue_ptr + score_size] = cls_score[score_ptr:]
            self.queue_ptrs[label] = queue_ptr + score_size

    def get_mean(self, c, default=0.7):
        queue_ptr = self.queue_ptrs[c]
        cached_logit = self.queue_logits[c, :queue_ptr]
        if len(cached_logit) == 0:
            return default
        else:
            mean = torch.mean(cached_logit)
            return float(mean)

    def get_mean_std(self, c, default=0.7):
        queue_ptr = self.queue_ptrs[c]
        cached_logit = self.queue_logits[c, :queue_ptr]
        if len(cached_logit) == 0:
            return default, 0.0
        else:
            mean = torch.mean(cached_logit)
            std = torch.std(cached_logit)
            return float(mean), float(std)


def get_tensor_length(tensor, size):
    length_tensor = tensor.new_tensor([len(tensor)])
    length_list = [tensor.new_tensor([0]) for _ in range(size)]
    dist.all_gather(length_list, length_tensor)

    lengths = torch.cat(length_list)
    max_length = int(lengths.max().cpu().numpy())
    current_length = len(tensor)

    return max_length, current_length


def collect_tensor_from_dist(tensor_list, fill_numbers):
    size = dist.get_world_size()

    if not isinstance(tensor_list, list):
        tensor_list = [tensor_list]
    max_length, current_length = get_tensor_length(tensor_list[0], size)

    tensor_collection = []
    for tensor, number in zip(tensor_list, fill_numbers):
        shape = list(tensor.shape)
        shape[0] = max_length
        gather_tensors = [tensor.new_zeros(size=shape) for _ in range(size)]

        shape = list(tensor.shape)
        shape[0] = max_length - shape[0]
        fill_tensor = torch.cat([tensor, tensor.new_ones(size=shape) * number])
        dist.all_gather(gather_tensors, fill_tensor)
        gather_tensors = torch.cat(gather_tensors)
        tensor_collection.append(gather_tensors)

    return tensor_collection
