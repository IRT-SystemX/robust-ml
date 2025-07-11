# Copy dependencies
cd ..
cp -R _static docs/source/
cp -R examples docs/source/

# Delete old robustML modules

rm -f docs/source/robustML*.rst

# Generate package docstring

sphinx-apidoc -o docs/source robustML

# Generate HTML

cd docs
./make.bat clean
./make.bat html

# Clean temp directories
rm -Rf source/_static
rm -Rf source/examples
