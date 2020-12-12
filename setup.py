from setuptools import find_packages, setup

setup(
    name='bjtunlp',
    version='1.0.0',
    author='Kai Wang',
    author_email='425065178@qq.com/19120406@bjtu.edu.cn',
    description='nlp toolkit for Chinese only',
    url='https://github.com/bosskai/bjtunlp',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic"
    ],
    install_requires=["fastnlp>=0.6.0", "transformers==3.1.0"],
    long_description='hello word',
    entry_points={
        'console_scripts': [
            'bjtunlp-train-joint=bjtunlp.train:main'
        ]
    },
    python_requires='>=3.7',
    zip_safe=False
)