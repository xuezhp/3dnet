# test plotting codes
import numpy as np
import dataIO as d
def plotvol(objname,is_local,obj_ratio):
	print(objname)
	volumes = d.getAll(obj=objname,train=True,is_local=is_local,obj_ratio=obj_ratio)
	volumes = volumes[...,np.newaxis].astype(np.float)
	ind = np.random.randint(len(volumes),size=1)
	x = volumes[ind]
	d.plotFromVoxels(x,'Voxel_'+'test'+'_'+'.png')
	d.plotMeshFromVoxels(x,threshold=0.5,filename='Mesh_'+'test'+'_'+'.png')


if __name__=="__main__":
	plotvol('chair',True,0.7)
