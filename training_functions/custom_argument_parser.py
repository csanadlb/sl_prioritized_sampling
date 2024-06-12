from argparse import ArgumentParser


class CustomArgParser:

    def __init__(self, config):
        self.config = config

    def get_parsed_config(self):
        arg_parser = ArgumentParser()

        arg_parser.add_argument("--n_epochs", type=int, default=self.config.getint("trainer", "n_epochs"))
        arg_parser.add_argument("--seed", type=int, default=self.config.getint("trainer", "seed"))
        arg_parser.add_argument("--prioritized_sampler", type=int,
                                default=self.config.getint("data_loader", "prioritized_sampler"))
        arg_parser.add_argument("--explore_type", type=str, default=self.config.get("data_loader", "explore_type"))
        arg_parser.add_argument("--exploit_type", type=str, default=self.config.get("data_loader", "exploit_type"))
        arg_parser.add_argument("--c_const", type=float, default=self.config.getfloat("data_loader", "c_constant"))
        arg_parser.add_argument("--batch_size_train", type=int,
                                default=self.config.getint("data_loader", "batch_size_train"))
        arg_parser.add_argument("--probs", type=float, default=self.config.getfloat("augmentation", "probs"))
        arg_parser.add_argument("--normalize", type=int, default=self.config.getint("augmentation", "normalize"))
        arg_parser.add_argument("--lr_min", type=float, default=self.config.getfloat("agent", "lr_min"))
        arg_parser.add_argument("--optimizer", type=str, default=self.config.get("agent", "optimizer"))
        arg_parser.add_argument("--warmup_epochs", type=int, default=self.config.getint("agent", "warmup_epochs"))
        arg_parser.add_argument("--warmup_lr_high", type=float, default=self.config.getfloat("agent", "warmup_lr_high"))
        arg_parser.add_argument("--momentum", type=float, default=self.config.getfloat("agent", "momentum"))
        arg_parser.add_argument("--weight_decay", type=float, default=self.config.getfloat("agent", "weight_decay"))
        arg_parser.add_argument("--beta1", type=float, default=self.config.getfloat("agent", "beta1"))
        arg_parser.add_argument("--beta2", type=float, default=self.config.getfloat("agent", "beta2"))

        parsed_args = arg_parser.parse_args()

        self.config.set("trainer", "n_epochs", str(parsed_args.n_epochs))
        self.config.set("trainer", "seed", str(parsed_args.seed))
        self.config.set("data_loader", "prioritized_sampler", str(parsed_args.prioritized_sampler))
        self.config.set("data_loader", "explore_type", str(parsed_args.explore_type))
        self.config.set("data_loader", "exploit_type", str(parsed_args.exploit_type))
        self.config.set("data_loader", "batch_size_train", str(parsed_args.batch_size_train))
        self.config.set("augmentation", "probs", str(parsed_args.probs))
        self.config.set("agent", "lr_min", str(parsed_args.lr_min))
        self.config.set("agent", "optimizer", str(parsed_args.optimizer))
        self.config.set("agent", "warmup_epochs", str(parsed_args.warmup_epochs))
        self.config.set("agent", "warmup_lr_high", str(parsed_args.warmup_lr_high))
        self.config.set("agent", "momentum", str(parsed_args.momentum))
        self.config.set("agent", "weight_decay", str(parsed_args.weight_decay))
        self.config.set("agent", "beta1", str(parsed_args.beta1))
        self.config.set("agent", "beta2", str(parsed_args.beta2))
        self.config.set("augmentation", "normalize", str(parsed_args.normalize))
        self.config.set("data_loader", "c_constant", str(parsed_args.c_const))

        return self.config
