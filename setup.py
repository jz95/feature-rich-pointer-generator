from setuptools import setup

setup(name='feature-rich-pg',
      version='1.0',
      description='An extension for the pointer generator network for abstractive text summerization',
      url='https://github.com/JZ95/feature-rich-pointer-generator',
      author='JZhou, ShiHao Liu, Christos Drou',
      author_email='j.zhou0518@gmail.com, liushihao0927@gmail.com, rhaegal1992@gmail.com',
      license='MIT',
      packages=['frpg', 'frpg.pos_tagger'],
      scripts=['./bin/frpg_run', './bin/frpg_eval_rouge'],
      install_requires=[
          'pyrouge',
          'nltk==3.3.0'
      ],
      zip_safe=False)