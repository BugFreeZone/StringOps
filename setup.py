from setuptools import setup, find_packages

setup(
    name='StringOps',
    version='1.0.0',
    description='Небольшая Python библиотека со строковыми операциями',
    author='BugFreeZone',
    author_email='r96177385@gmail.com',
    packages=find_packages(),
    install_requires=[
        'chardet'
    ],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BugFreeZone/StringOps',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
