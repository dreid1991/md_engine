################################################
#  Simplest pytest example, taken from somewhere
#
#  pytest searches for functions, classes etc.
#  prefixed by 'test_'
#
#  Learn more about pytest by reading the docs
#  at docs.pytest.org
################################################

def func(x):
    return x + 1

def test_answer():
    assert func(4) == 5
