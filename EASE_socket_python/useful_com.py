def dump(x):
    return ''.join([type(x).__name__, "('",
                    *['\\x'+'{:02x}'.format(i) for i in x], "')"])
