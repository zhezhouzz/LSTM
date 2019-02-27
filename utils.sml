fun intpow a b =
    if b = 0 then 1 else a * (intpow a (b - 1));
