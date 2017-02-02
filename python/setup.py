from distutils.core import setup, Extension

incDirsPy = '/home/danielreid/md_engine/core/src;/home/danielreid/md_engine/core/src/GPUArrays;/home/danielreid/md_engine/core/src/Integrators;/home/danielreid/md_engine/core/src/Fixes;/home/danielreid/md_engine/core/src/DataStorageUser;/home/danielreid/md_engine/core/src/Evaluators;/home/danielreid/md_engine/core/src/BondedForcers'.split(';')

module1 = Extension('Sim',
                    sources = ['/home/danielreid/md_engine/core/python/Sim.cpp'],
                    library_dirs = ["/home/danielreid/md_engine/core"],
                    libraries = ["Sim"],
                    include_dirs = incDirsPy + 
                                    ["/software/cuda-7.5-el7-x86_64/include"],
                    runtime_library_dirs = ["/usr/local/lib"],
                    extra_compile_args = " -std=c++11 -fpic".split()
                    )

setup(name='Sim',
      version='0.4',
      description='A GPU-base Molecular Dynamics simulation engine',
      author='Daniel Reid',
      author_email='danielreid@uchicago.edu',
      package_dir={ '': '/home/danielreid/md_engine/core/python' },
      ext_modules=[module1],
      )
