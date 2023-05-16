class Flags:
    def __init__(self, s=''):
        self.flags = s

    def add_definition(self, key, value):
        if isinstance(value, bool):
            value = int(value)
        self.flags += ' -D%s=%s' % (key, str(value))
        return self

    def add_string(self, s):
        self.flags += ' %s' % str(s)
        return self

    def __str__(self):
        return self.flags
