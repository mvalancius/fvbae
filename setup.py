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
  download_url = 'https://github.com/mvalancius/vbae/archive/v_02.tar.gz',    # I explain this later on
  keywords = ['Autoencoder', 'MNIST', 'Variational'],   
  install_requires=[            # I get to this in a second
          'numpy',
          'keras',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
