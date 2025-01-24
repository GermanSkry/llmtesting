# src/__init__.py
import importlib
import pkgutil

# Initialize __all__ as an empty list
__all__ = []

# Get the current package name
package_name = "src"

# Import modules one by one (avoiding dynamic imports of everything at once)
module_names = ['embeddings','file_uploader', 'embedding_database', 'Rag_preprocess', 'rag']
for module_name in module_names:
    module = importlib.import_module(f"{package_name}.{module_name}")
    
    # Add the module functions to globals
    for attr in dir(module):
        if not attr.startswith("_"):  # Ignore private attributes
            globals()[attr] = getattr(module, attr)

    # Register the module name in __all__
    __all__.append(module_name)