

def test_mayavi():
    # check if mayavi is the bad guy on the CI
    print('Importing mayavi...')
    from mayavi import mlab
    print('Done.')
