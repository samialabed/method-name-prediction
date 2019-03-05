import json
from typing import Any, Dict

from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.engine.saving import load_model


class BaseModel(Model):

    def __init__(self, hyperparameters: Dict[str, Any]):
        super(BaseModel, self).__init__()
        self.hyperparameters = hyperparameters
        self.model = Sequential()

    def predict_name(self, code_block: str):
        raise NotImplementedError

    @staticmethod
    def from_file(path: str):
        """
        :arg path directory path to a file that contains, config, model and weights.
        :return a model populated from a file path.
        """
        return load_model('{}/model.h5'.format(path))

    def save(self, filepath, overwrite=True, include_optimizer=True) -> None:
        self.model.save_weights(filepath)
        model_type = type(self).__name__
        model_config_to_save = {
            "model_type": model_type,
            "hyperparameters": self.hyperparameters,
        }

        # Save hyperparameters
        with open('{path}/{name}/model_config.json'.format(path=filepath, name=model_type)) as fp:
            json.dump(model_config_to_save, fp)

        # Save the model architecture
        with open('{path}/{name}/model.json'.format(path=filepath, name=model_type)) as model_json:
            model_json.write(self.model.to_json())

        # Save the weight
        self.model.save_weights('{path}/{name}/model_weights.h5'.format(path=filepath, name=model_type))

        # Save the model completely
        self.model.save('{path}/{name}/model.h5'.format(path=filepath, name=model_type))
