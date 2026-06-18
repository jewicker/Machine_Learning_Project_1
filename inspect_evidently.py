import importlib.util
import evidently
print('evidently version:', evidently.__version__)
print('evidently file:', evidently.__file__)
for module in ['evidently.model_profile', 'evidently.model_profile.model_profile', 'evidently.model_profile.sections']:
    print(module, importlib.util.find_spec(module))
