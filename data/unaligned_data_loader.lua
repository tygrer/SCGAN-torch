--------------------------------------------------------------------------------
-- Subclass of BaseDataLoader that provides data from two datasets.
-- The samples from the datasets are not aligned.
-- The datasets can have different sizes
--------------------------------------------------------------------------------

require 'cutorch'
require 'cunn'
require 'data.base_data_loader'
local matio = require 'matio'
local class = require 'class'
data_util = paths.dofile('data_util.lua')

UnalignedDataLoader = class('UnalignedDataLoader', 'BaseDataLoader')

function UnalignedDataLoader:__init(conf)
  BaseDataLoader.__init(self, conf)
  conf = conf or {}
end

function UnalignedDataLoader:name()
  return 'UnalignedDataLoader'
end

function UnalignedDataLoader:Initialize(opt)
  opt.align_data = 0
  self.dataA = data_util.load_dataset('A', opt, opt.input_nc)
  -- print(opt.output_nc)
  self.dataB = data_util.load_dataset('B', opt, opt.output_nc)
end

-- actually fetches the data
-- |return|: a table of two tables, each corresponding to
-- the batch for dataset A and dataset B
function UnalignedDataLoader:LoadBatchForAllDatasetstrain()
  local batchA, pathA = self.dataA:getBatch()
--  print(pathA[1])
--  print(string.len(pathA[1]))
  local lapstrA=string.sub(pathA[1],54,string.len(pathA[1]))
--print(lapstrA)
  local CSR_fnA = '/home/tgy/CycleBIG/result/Input_Laplacian_'..  lapstrA .. '.mat'
--  print('loading matting laplacian...', CSR_fnA)
  local CSRA = matio.load(CSR_fnA).CSR:cuda()
--  print(type(CSRA))
  local batchB, pathB = self.dataB:getBatch()
  local lapstrB=string.sub(pathB[1],54,string.len(pathB[1]))
--  print(lapstrB)
  local CSR_fnB = '/home/tgy/CycleBIG/result/Input_Laplacian_'..  lapstrB .. '.mat'
--  print('loading matting laplacian...', CSR_fnB)
  local CSRB = matio.load(CSR_fnB).CSR:cuda()
--  print('batch_pathB', pathB)
  return batchA, batchB, pathA, pathB, CSRA, CSRB
end

function UnalignedDataLoader:LoadBatchForAllDatasetstest()
  local batchA, pathA = self.dataA:getBatch()
--  print(pathA[1])
--  print(string.len(pathA[1]))
  local lapstrA=string.sub(pathA[1],50,string.len(pathA[1]))
--print(lapstrA)
--  print(type(CSRA))
  local batchB, pathB = self.dataB:getBatch()
  local lapstrB=string.sub(pathB[1],50,string.len(pathB[1]))
--  print(lapstrB)
 
--  print('batch_pathB', pathB)
  return batchA, batchB, pathA, pathB
end
-- returns the size of each dataset
function UnalignedDataLoader:size(dataset)
  if dataset == 'B' then
  print ('dataBsize:',self.dataB:size())
    return self.dataB:size()
  end

  -- return the size of the first dataset by default
  return self.dataA:size()
end
