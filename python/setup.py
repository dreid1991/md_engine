from distutils.core import setup, Extension

incDirsPy = '/home/daniel/Documents/md_engine/core/src;/home/daniel/Documents/md_engine/core/src/GPUArrays;/home/daniel/Documents/md_engine/core/src/Integrators;/home/daniel/Documents/md_engine/core/src/Fixes;/home/daniel/Documents/md_engine/core/src/DataStorageUser;/home/daniel/Documents/md_engine/core/src/Evaluators;/home/daniel/Documents/md_engine/core/src/BondedForcers'.split(';')

module1 = Extension('Sim',
                    sources = ['/home/daniel/Documents/md_engine/core/python/Sim.cpp'],
                    library_dirs = ["/home/daniel/Documents/md_engine/core/src"],
                    libraries = ["Sim"],
                    include_dirs = incDirsPy + 
                                    ["/usr/local/cuda-7.5/include"],
                    runtime_library_dirs = ["/usr/local/lib"],
                    extra_compile_args = " -std=c++11 -fpic".split()
                    )

setup(name='Sim',
      version='0.4',
      description='A GPU-base Molecular Dynamics simulation engine',
      author='Daniel Reid',
      author_email='danielreid@uchicago.edu',
      package_dir={ '': '/home/daniel/Documents/md_engine/core/python' },
      ext_modules=[module1],
      )
