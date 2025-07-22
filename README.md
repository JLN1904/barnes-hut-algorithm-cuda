Codigo creado por Jose Luis Novoa y Juan Binimelis.

# Proyecto de implementaciones secuencial y paralela del algoritmo Barnes-Hut

## Uso
    Para ajustar el tamaño de la simulación hay que editar la variable N, ubicada al principio de la funcion main de los archivos:
    - main.cu en barnes_cuda para la implementación paralela
    - main.cpp en barnes_sec para la implementación secuencial

### Uso Secuencial
    En la carpeta barnes_sec
    '''
    g++ main.cpp -o <nombre>
    ./<nombre>
    '''

### Uso Paralela
    En la carpeta barnes_cuda
    '''
    make clean
    make
    ./nbody_cuda
    '''


### Consideraciones
    - El archivo Makefile fue creado para GPUs con arquitectura "sm_86" (NVIDIA serie 3000).
    - En caso de no poseer una GPU con dicha arquitectura, ejecutar las siguientes lineas en vez del Makefile:
    '''
    nvcc -gencode arch=compute_XX,code=sm_XX -gencode arch=compute_XX,code=compute_XX -I/usr/local/cuda/include -rdc=true -c main.cu -o main.o
    nvcc -gencode arch=compute_XX,code=sm_XX -gencode arch=compute_XX,code=compute_XX -I/usr/local/cuda/include -rdc=true -c nbody_cuda_kernels.cu -o nbody_cuda_kernels.o
    nvcc -gencode arch=compute_XX,code=sm_XX -gencode arch=compute_XX,code=compute_XX -I/usr/local/cuda/include -rdc=true -c quadtree_gpu.cu -o quadtree_gpu.o
    nvcc -gencode arch=compute_XX,code=sm_XX -gencode arch=compute_XX,code=compute_XX -I/usr/local/cuda/include -rdc=true -c plummer.cpp -o plummer.o
    nvcc -gencode arch=compute_XX,code=sm_XX -gencode arch=compute_XX,code=compute_XX -I/usr/local/cuda/include -rdc=true -arch=sm_XX main.o nbody_cuda_kernels.o quadtree_gpu.o plummer.o -o nbody_cuda -L/usr/local/cuda/lib64 -lcudart
    ./nbody_cuda
    '''
    - Reemplazar todos los "XX" por la arquitectura correspondiente (Consultar https://developer.nvidia.com/cuda-gpus para una GPU especifica).    

