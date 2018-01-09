# test plotting codes
import numpy as np
import dataIO as d
def plotvol(objname,is_local,obj_ratio):
	print(self.objname)
	volumes = d.getAll(obj=objname,train=True,is_local=is_local,obj_ratio=obj_ratio)
	volumes = volumes[...,np.newaxis].astype(np.float)
	ind = np.random.randint(len(volumes),1)
	x = volumes[ind]
	d.plotFromVoxels(np.squeeze(g_obj[id_ch[i]]>0.5),'Voxel_'+str(epoch)+'_'+str(i)+'.png')
	d.plotMeshFromVoxels(np.squeeze(g_obj[id_ch[i]]>0.5),threshold=0.5,filename='Mesh_'+str(epoch)+'_'+str(i)+'.png')


if __name__=="__main__":
	plotvol('chair',True,0.7)
