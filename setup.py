import setuptools

setuptools.setup(
    name='clustering_benchmark',
    packages=['clustering_benchmark'],
    version='0.1',
    license='MIT',
    author='Yuanjun Xiong',
    author_email='bitxiong@gmail.com',
    description='Simple benchmarking for clustering algorithms.',
    long_description='This benchmark implements basic clustering benchmarks including metrics like V-measure.',
    url='https://https://github.com/yjxiong/clustering-benchmark',
    download_url='',
    keywords=['clustering', 'benchmark'],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)