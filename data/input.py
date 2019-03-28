from options.options import Options
import torch


def get_attr_name(opt):
    filename = "%s/Anno/list_attr_cloth.txt" % opt.data_dir
    f = open(filename)
    num_files = int(f.readline())
    f.readline()
    attrs_name = []
    attrs_type = torch.zeros(num_files)
    i=0
    for line in f:
       word = line.strip()[:-1].strip()
       word2=line.strip()[-1]
       attrs_name.append(word)
       attrs_type[i] = float(word2)
       i=i+1
    f.close()
    return attrs_name, attrs_type

def get_Ctg_name(opt):
    filename = "%s/Anno/list_category_cloth.txt" % opt.data_dir
    f = open(filename)
    f.readline()
    f.readline()
    Ctg_name=[]
    Ctg_type=[]
    for line in f:
       word = line.strip()[:-1].strip()
       word2 = line.strip()[-1]
       Ctg_name.append(word)
       Ctg_type.append(word2)

    f.close()
    return Ctg_name, Ctg_type


if __name__ == "__main__":
   op = Options()
   opt = op.parse()
   attr_name,attr_type = get_attr_name(opt)
