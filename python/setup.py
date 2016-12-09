from distutils.core import setup, Extension

incDirsPy = '/home/danielreid/md_engine_ssages/core/src;/home/danielreid/md_engine_ssages/core/src/GPUArrays;/home/danielreid/md_engine_ssages/core/src/Integrators;/home/danielreid/md_engine_ssages/core/src/Fixes;/home/danielreid/md_engine_ssages/core/src/DataStorageUser;/home/danielreid/md_engine_ssages/core/src/Evaluators;/home/danielreid/md_engine_ssages/core/src/BondedForcers'.split(';')

module1 = Extension('Sim',
                    sources = ['/home/danielreid/md_engine_ssages/core/python/Sim.cpp'],
                    library_dirs = ["/home/danielreid/md_engine_ssages/core"],
                    libraries = ["Sim"],
                    include_dirs = incDirsPy + 
                                    ["/software/cuda-8.0-el6-x86_64/include"],
                    runtime_library_dirs = ["/home/danielreid/ssages_gpu_midway1/build/danmd/lib"],
                    extra_compile_args = " -std=c++11 -fpic".split()
                    )

setup(name='Sim',
      version='0.4',
      description='A GPU-base Molecular Dynamics simulation engine',
      author='Daniel Reid',
      author_email='danielreid@uchicago.edu',
      package_dir={ '': '/home/danielreid/md_engine_ssages/core/python' },
      ext_modules=[module1],
      )
