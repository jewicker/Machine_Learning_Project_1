import traceback
try:
    import app
    print('IMPORT_OK')
except Exception as e:
    traceback.print_exc()
    print('IMPORT_FAILED')
