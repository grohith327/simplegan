from distutils.core import setup

setup(
  name = 'easygan',         
  packages = ['easygan'],   
  version = '0.1',    
  license='MIT',      
  description = 'Framework to ease training of generative models based on TensorFlow',   
  author = 'Rohith Gandhi G',             
  author_email = 'grohith327@gmail.com',  
  url = 'https://github.com/grohith327',  
  download_url = 'https://github.com/grohith327/EasyGAN/archive/v0.1.tar.gz',    # I explain this later on
  keywords = ['GAN', 'Computer Vision', 'Deep Learning', 'TensorFlow', 'Generative Models', 'Neural Networks', 'AI'],
  install_requires=[            
          'tensorflow',
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
    'Topic :: Scientific/Engineering :: Deep Learning',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)