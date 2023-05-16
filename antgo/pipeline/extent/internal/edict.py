"""easy dict"""


class edict(dict):
    """a dict class which could be accessed by attribute"""

    def __setattr__(self, key, value):
        super(edict, self).__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return super(edict, self).__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __delattr__(self, key):
        super(edict, self).__delitem__(key)
