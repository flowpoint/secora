import ray
from ray import tune

raise NotImplementedError()


ray.init(local_mode=True)

analysis = tune.run(
        training_run,
        config=config,
        local_dir=config['logdir'],
        metric='mrr',
        mode='max',
        num_samples=1,
        keep_checkpoints_num=10,
        name=config['name'],
        resume="AUTO"
        #resources_per_trial={'cpu':10},
        )

dfs = analysis.trial_dataframes

class TuneParamTracker(ParamTracker):
    def __init__(self, config):
        super().__init__(self,config)

    def save(self, model, optimizer, **kwargs):
        with tune.checkpoint_dir(k) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optim.state_dict()), path)

    def load(self, model, optimizer, checkpoint="", **kwargs):
        path = os.path.join(self.config['checkpoint_dir'], "checkpoint")
        params = (model.state_dict(), optim.state_dict()) + kwargs.items()
        torch.save(params, path)

        model_state, optimizer_state, other_params = torch.load(path)
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

        return model, optim, other_params


