from typing import List
from keras.models import Sequential
from keras import layers
from keras.regularizers import L2
from common.model import Base

class NeuMF(Base):
    def __init__(self, user_dim: int, item_dim: int, output_dim: int = 8, nums_hiddens: List[int] = [8, 8, 8], *args, **kwargs):
        super(NeuMF, self).__init__(*args, **kwargs)
        # GMF Embedding
        self.P = layers.Embedding(input_dim=user_dim, output_dim=output_dim, embeddings_regularizer=L2(0.001))
        self.Q = layers.Embedding(input_dim=item_dim, output_dim=output_dim, embeddings_regularizer=L2(0.001))

        # MLP Embedding
        self.U = layers.Embedding(input_dim=user_dim, output_dim=output_dim, embeddings_regularizer=L2(0.001))
        self.V = layers.Embedding(input_dim=item_dim, output_dim=output_dim, embeddings_regularizer=L2(0.001))

        # GMF Layer
        self.element_wise = layers.Multiply()

        # MLP Layer
        self.mlp_concat = layers.Concatenate(axis=1)
        self.mlp = Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(layers.Dense(num_hiddens, activation="relu"))

        # Concatenate Layer
        self.concat_layer = layers.Concatenate(axis=1)

        # Predict Layer
        self.predict_layer = layers.Dense(
            1, activation="sigmoid", use_bias=False, kernel_regularizer=L2(0.001))

    def call(self, inputs, training=None, mask=None):
        # GMF Layer
        p_mf = self.P(inputs[0])
        q_mf = self.Q(inputs[1])

        gmf = self.element_wise([p_mf, q_mf])

        # MLP Layer
        p_mlp = self.U(inputs[0])
        q_mlp = self.V(inputs[1])

        mlp = self.mlp(self.mlp_concat([p_mlp, q_mlp]))

        concat_result = self.concat_layer([gmf, mlp])
        concat_result = layers.Flatten()(concat_result)
        return self.predict_layer(concat_result)
