
# MICCoM Demo

The point of this demo is to demonstrate a high-level workflow script
that can drive multiple concurrent runs of **gb** as part of an
ensemble.

## Build

This example needs the `Sim.so` library.  From the top level:

```
make lib
```

This builds the CUDA modules and the Python interface.

## Python

The demo Python use case from Reid is in `pvd.py`.  It was refactored
to make it importable and callable as a function (`pvd()`) from other
Python code (and thus from Swift/T).

You can run it with:

```
pvd-run.sh
```

## Swift/T

We can call `pvd()` from Swift/T.  Run this with:

```
pvd-swift.sh
```

Add `/opt/swift-t/stc/bin` to your `PATH` to get the `swift-t`
program.

<!--- COMMENT -->