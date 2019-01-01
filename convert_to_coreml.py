from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import coremltools

filename = 'models/model-med'

model = load_model(filename + '.h5')
coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save(filename + '.mlmodel')


