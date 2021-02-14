import gc
import torch

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def restore_model(model_class, load_path, model_args_extras=None):
    """
    Restores a model from the given path. `model_class` must be the class for
    which the saved model was created from. This will create a model of this
    class, using the loaded creation arguments. It will then restore the learned
    parameters to the model.
    """
    load_dict = torch.load(load_path)
    model_state = load_dict["model_state"]
    model_creation_args = load_dict["model_creation_args"]
    if model_args_extras is not None:
        model_creation_args.update(model_args_extras)
    # print(model_creation_args) ####
    model = model_class(**model_creation_args)
    model.load_state_dict(model_state)
    return model

def torch_mem_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass