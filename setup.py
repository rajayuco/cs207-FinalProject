import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
                 name="autodiffpy",
                 version="0.0.1",
                 author="Rachel Moon, Cory Williams, Yudi Wang, Jamila Pegues",
                 author_email="rachelmoon@g.harvard.edu, cwilliams@g.harvard.edu, yudiwang@g.harvard.edu, jpegues@g.harvard.edu",
                 description="Autodifferentiation software package",
                 url="https://github.com/rajayuco/cs207-FinalProject",
                 packages=setuptools.find_packages(),
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "Operating System :: OS Independent",
                              ],
                 )
