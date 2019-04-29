from distutils.core import setup
setup(
  name = 'fvbae',         
  packages = ['fvbae'],   
  version = '0.1',      
  license='MIT',        
  description = 'Keras Implementation of Variable Bayesian Autoencoder (Kingma and Welling, 2014)',  
  author = 'Michael Valancius, Evan Poworoznek',                   
  author_email = 'mvalancius18@gmail.com',      
  url = 'https://github.com/mvalancius',   
  download_url = 'https://github.com/mvalancius/fvbae/archive/v_01.tar.gz',    
  keywords = ['Autoencoder', 'MNIST', 'Variational'],   
  install_requires=[            
          'numpy',
          'keras',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
