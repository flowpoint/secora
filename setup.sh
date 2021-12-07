apt update && apt install 

git clone https://github.com/facebookresearch/faiss

# configure
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=OFF -B build .

# build
make -C build -j faiss

# build python bindings
make -C build -j swigfaiss

# install python faiss
(cd build/faiss/python && python setup.py install)

# run python tests
(cd build/faiss/python && python setup.py build)
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py

