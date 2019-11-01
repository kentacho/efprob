import setuptools

setuptools.setup(
    name='efprob',
    version='0.1',
    description='Channel-based probability calculations',
    author='Bart Jacobs, Kenta Cho',
    author_email='bart@cs.ru.nl, cho@nii.ac.jp',
    packages=['efprob'],
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'matplotlib'],
    extras_require={
        'test': ['pytest']
    }
)
