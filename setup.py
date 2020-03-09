import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
  name = 'simplegan',         
  packages = find_packages(),   
  version = '0.2.6',    
  license='MIT',      
  description = 'Framework to ease training of generative models based on TensorFlow',
  long_description=README,
  long_description_content_type="text/markdown",   
  author = 'Rohith Gandhi G',             
  author_email = 'grohith327@gmail.com', 
  url = 'https://github.com/grohith327',
  include_package_data=True,  
  keywords = ['GAN', 'Computer Vision', 'Deep Learning', 'TensorFlow', 'Generative Models', 'Neural Networks', 'AI'],
  install_requires=[            
          'tensorflow==2.0.1',
          'tqdm',
          'numpy',
          'opencv-python',
          'imageio',
          'tensorflow-datasets'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)