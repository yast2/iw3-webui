# Reference assemblies

The extension is compiled against four of Cove's own assemblies. They aren't
included here, so copy them out of your own Cove container:

```sh
docker cp Cove:/opt/cove/Cove.Sdk.dll     .
docker cp Cove:/opt/cove/Cove.Plugins.dll .
docker cp Cove:/opt/cove/Cove.Core.dll    .
docker cp Cove:/opt/cove/Cove.Data.dll    .
```

Take them from the **running container**, not from Cove's source code. A
released Cove usually lags behind its main branch, and the extension
interface differs between the two. Build against the newer one and you get
an extension that loads fine and then fails on its first call.

They're only needed for compiling (`<Private>false</Private>`), so nothing is
copied into the build output. Cove provides them when the extension runs.
