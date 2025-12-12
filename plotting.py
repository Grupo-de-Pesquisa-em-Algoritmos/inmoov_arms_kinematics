import roboticstoolbox as rtb
panda = rtb.models.URDF.Panda()
print(panda)
panda.plot(panda.qz, backend="swift")