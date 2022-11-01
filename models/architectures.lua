require 'nngraph'
require 'models/DenseConnectLayer'
require 'nn'
require 'cunn'
require 'cudnn'
----------------------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

nChannels=0
normalization = nil

function set_normalization(norm)
if norm == 'instance' then
  require 'util.InstanceNormalization'
  print('use InstanceNormalization')
  normalization = nn.InstanceNormalization
elseif norm == 'batch' then
  print('use SpatialBatchNormalization')
  normalization = nn.SpatialBatchNormalization
end
end

function defineG(input_nc, output_nc, ngf, which_model_netG, growthRate,bottomneck,droprate)
  local netG = nil
  if     which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
  elseif which_model_netG == "unet128" then netG = defineG_unet128(input_nc, output_nc, ngf)
  elseif which_model_netG == "unet256" then netG = defineG_unet256(input_nc, output_nc, ngf)
  elseif which_model_netG == "resnet_6blocks" then netG = defineG_resnet_6blocks(input_nc, output_nc, ngf)
  elseif which_model_netG == "resnet_9blocks" then netG = defineG_resnet_9blocks(input_nc, output_nc, ngf)
  elseif which_model_netG == "resnet_basic" then netG = defineG_resnet_basic(input_nc, output_nc, ngf)
  elseif which_model_netG == "densnet_6blocks" then
  netG = defineG_densnet_6blocks(input_nc, output_nc, ngf,bottomneck,droprate,growthRate)
  
  else error("unsupported netG model")
    end
  netG:apply(weights_init)
  
  return netG
end

function defineD(input_nc, ndf, which_model_netD,n_layers_D, use_sigmoid,bottomneck,droprate,growthRate)
  local netD = nil
  if     which_model_netD == "basic" then netD = defineD_basic(input_nc, ndf, use_sigmoid)
  elseif which_model_netD == "imageGAN" then netD = defineD_imageGAN(input_nc, ndf, use_sigmoid)
  elseif which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc, ndf, n_layers_D, use_sigmoid)
  elseif which_model_netD == "densnetD" then
  netD = defineD_dense(input_nc,ndf,n_layers_D, use_sigmoid,bottomneck,droprate,growthRate)
  else error("unsupported netD model")
  end
  netD:apply (weights_init)

  return netD
end

function defineG_encoder_decoder(input_nc, output_nc, ngf)
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- normalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf)
    -- input is (ngf) x128 x 128
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    local o1 = d8 - nn.Tanh()

    local netG = nn.gModule({e1},{o1})
    return netG
end


function defineG_unet128(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 128 x 128
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 64 x 64
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 32 x 32
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 16 x 16
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- normalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e6} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e5} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e4} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e3} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e2} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e1} - nn.JoinTable(2)

    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 128 x 128

    local o1 = d7 - nn.Tanh()
    local netG = nn.gModule({e1},{o1})
    return netG
end


function defineG_unet256(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- - normalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256

    local o1 = d8 - nn.Tanh()
    local netG = nn.gModule({e1},{o1})
    return netG
end

--------------------------------------------------------------------------------
-- Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
--------------------------------------------------------------------------------

local function build_conv_block(dim, padding_type)
  local conv_block = nn.Sequential()
  local p = 0
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  elseif padding_type == 'zero' then
    p = 1
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(normalization(dim))
  conv_block:add(nn.ReLU(true))
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(normalization(dim))
  return conv_block
end
function addLayer(model, nChannels, bottomneck,droprate,grownrate)
     model:add(DenseConnectLayerStandard(nChannels, bottomneck,droprate,grownrate))
   end


function addTransition(model, nChannels, nOutChannels, droprate, last, pool_size)
      --if opt.optMemory >= 3 then     
      --model:add(nn.JoinTable(2))
      --end
	  
      model:add(normalization(nChannels))
      model:add(nn.ReLU(true))      
      model:add(nn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      print('dropout',droprate)
	  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))	
     -- model:add(normalization(nOutChannels))
      --model:add(nn.ReLU(true))   
		 if droprate > 0 then 
		 model:add(nn.Dropout(droprate)) end
     
      --model:add(nn.SpatialAveragePooling(2, 2))
	  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
         --model:add(nn.SpatialAveragePooling(2, 2))
	     
end
local function addDenseBlock(model,nChannels,N,bottomneck,droprate,grownrate,last,pool_size)
	--local model = nn.Sequential()
	for i = 1, N do 
         addLayer(model, nChannels, bottomneck,droprate,grownrate)
         nChannels = nChannels + grownrate
    end
	addTransition(model, nChannels, math.floor(nChannels*0.5),droprate,last,pool_size)
    nChannels = math.floor(nChannels*0.5)
    return nChannels   
	
end

local function build_res_block(dim, padding_type)
  local conv_block = build_conv_block(dim, padding_type)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)
  if padding_type == 'none' or padding_type == 'reflect-start' then
    concat:add(nn.ShaveImage(2))
  else
    concat:add(nn.Identity())
  end
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end
function addDeposeLayer(model, nChannels,bottomneck,droprate,grownrate)
     model:add(DenseConnectLayerDeposeStandard(nChannels, bottomneck,droprate,grownrate))
   end


function addDeposeTransition(model, nChannels, droprate, last, pool_size)
      --if opt.optMemory >= 3 then     
      --model:add(nn.JoinTable(2))
      --end
	  
      model:add(normalization(nChannels))
      model:add(nn.ReLU(true))      
      model:add(nn.SpatialFullConvolution(nChannels, nChannels*2, 1, 1, 1, 1, 0, 0))
      print('dropout',droprate)
	  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))	
     -- model:add(normalization(nOutChannels))
      --model:add(nn.ReLU(true))   
		 if droprate > 0 then 
		 model:add(nn.Dropout(droprate)) end
      --model:add(nn.SpatialAveragePooling(2, 2))
	  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
         --model:add(nn.SpatialAveragePooling(2, 2))
	     
end
local function addDeposeDenseBlock(model,nChannels,N,bottomneck,droprate,grownrate,last,pool_size)
--local model = nn.Sequential()
	for i = 1, N do 
         addDeposeLayer(model, nChannels, bottomneck,droprate,grownrate)
         nChannels = nChannels + grownrate
    end
	addDeposeTransition(model, nChannels,droprate,last,pool_size)
    nChannels = math.floor(nChannels*2)
    return nChannels    
end
function defineG_densnet_6blocks(input_nc, output_nc, ngf,bottomneck,droprate,grownrate)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local model = nn.Sequential()
  model:add(nn.SpatialReflectionPadding(p, p, p, p))
  model:add(nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1))
  model:add(normalization(ngf))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1))
  model:add(normalization(ngf*2))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1))
  model:add(normalization(ngf*4))
  model:add(nn.ReLU(true))
  nChannels=ngf*4
  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  nChannels = addDenseBlock(model,ngf*4,2,bottomneck,droprate,grownrate)
  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  nChannels = addDenseBlock(model,nChannels,2,bottomneck,droprate,grownrate)
  
  --nChannels = addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate)
 
  --nChannels = addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate)

  --nChannels = addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate,false)
   --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  nChannels = addDeposeDenseBlock(model,nChannels,2,bottomneck,droprate,grownrate)
 -- model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  nChannels = addDeposeDenseBlock(model,nChannels,2,bottomneck,droprate,grownrate,true)
  --model:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  --nChannels = addDeposeDenseBlock(model,nChannels,1,bottomneck,droprate,grownrate,true)
  model:add(nn.SpatialFullConvolution(nChannels, ngf*2, ks, ks, 2, 2, 1, 1,1,1))
  model:add(normalization(ngf*2))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1))
  model:add(normalization(ngf))
  model:add(nn.ReLU(true))
  model:add(nn.SpatialReflectionPadding(p, p, p, p))
  model:add(nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1))
  model:add(nn.Tanh())
  return model
end
function defineG_resnet_6blocks(input_nc, output_nc, ngf)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local data = -nn.Identity()
  local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
  local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
  local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
  netG = nn.gModule({data},{d4})
  return netG
end
function defineG_resnet_basic(input_nc, output_nc, ngf)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local data = -nn.Identity()
  local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
  local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) 
  local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
  local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
  netG = nn.gModule({data},{d4})
  return netG
end
function defineG_resnet_9blocks(input_nc, output_nc, ngf)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local data = -nn.Identity()
  local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
  local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
  local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
  netG = nn.gModule({data},{d4})
  return netG
end

function defineD_imageGAN(input_nc, ndf, use_sigmoid)
    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 64 x 64
    netD:add(nn.SpatialConvolution(ndf * 2, ndf*4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 32 x 32
    netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 16 x 16
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 8 x 8
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
    netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4, 2, 2, 1, 1))
    -- state size: 1 x 1 x 1
    if use_sigmoid then
      netD:add(nn.Sigmoid())
    end

	return netD
end



function defineD_basic(input_nc, ndf, use_sigmoid)
    n_layers = 3
    return defineD_n_layers(input_nc, ndf, n_layers, use_sigmoid)
end

-- rf=1
function defineD_pixelGAN(input_nc, ndf, use_sigmoid)

    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
    netD:add(normalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    if use_sigmoid then
      netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
    end

    return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, ndf, n_layers, use_sigmoid, kw, dropout_ratio)

  if dropout_ratio == nil then
    dropout_ratio = 0.0
  end

  if kw == nil then
	kw = 4
  end
  padw = math.ceil((kw-1)/2)

    if n_layers==0 then
        return defineD_pixelGAN(input_nc, ndf, use_sigmoid)
    else

        local netD = nn.Sequential()

        -- input is (nc) x 256 x 256
        -- print('input_nc', input_nc)
        netD:add(nn.SpatialConvolution(input_nc, ndf, kw, kw, 2, 2, padw, padw))
        netD:add(nn.LeakyReLU(0.2, true))

        local nf_mult = 1
        local nf_mult_prev = 1
        for n = 1, n_layers-1 do
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, kw, kw, 2, 2, padw,padw))
            netD:add(normalization(ndf * nf_mult)):add(nn.Dropout(dropout_ratio))
            netD:add(nn.LeakyReLU(0.2, true))
        end

        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, kw, kw, 1, 1, padw, padw))
        netD:add(normalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, kw, kw, 1, 1, padw,padw))
        -- state size: 1 x (N-2) x (N-2)
        if use_sigmoid then
          netD:add(nn.Sigmoid())
        end
        -- state size: 1 x (N-2) x (N-2)
        return netD
    end
end
  



function defineD_dense(input_nc, ndf, n_layers, use_sigmoid,bottomneck,droprate,growthRate)
  
  if dropout_ratio == nil then
    dropout_ratio = 0.0
  end

  if kw == nil then
	kw = 4
  end
  padw = math.ceil((kw-1)/2)

    if n_layers==0 then
        return defineD_pixelGAN(input_nc, ndf, use_sigmoid)
    else

        local netD = nn.Sequential()

        -- input is (nc) x 256 x 256
        -- print('input_nc', input_nc)
        netD:add(nn.SpatialConvolution(input_nc, ndf, kw, kw, 2, 2, padw, padw))
        netD:add(nn.LeakyReLU(0.2, true))
        --local nf_mult = 1
        --local nf_mult_prev = 1
        --for n = 1, n_layers-1 do
            --nf_mult_prev = nf_mult
            --nf_mult = math.min(2^n,8)
            --netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, kw, kw, 2, 2, padw,padw))
           -- netD:add(normalization(ndf * nf_mult)):add(nn.Dropout(dropout_ratio))
            --netD:add(nn.LeakyReLU(0.2, true))
        --end
		--nChannels = addDenseBlockD(model, ndf, 5,bottomneck,droprate,grownrate)
		--addTransitionD(model, nChannels, math.floor(nChannels*reduction), opt)
        --nChannels = math.floor(nChannels*reduction)

       nChannels=addDenseBlock(netD,ndf,5,bottomneck,droprate,growthRate)
	  
       nChannels=addDenseBlock(netD,nChannels,5,bottomneck,droprate,growthRate)
       --addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate)
       --addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate) 
	   --addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate)
       --addDenseBlock(model,nChannels,5,bottomneck,droprate,grownrate) 
        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(nChannels, ndf * nf_mult, kw, kw, 1, 1, padw, padw))
        netD:add(normalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, kw, kw, 1, 1, padw,padw))
        -- state size: 1 x (N-2) x (N-2)
        if use_sigmoid then
          netD:add(nn.Sigmoid())
        end
        -- state size: 1 x (N-2) x (N-2)
        return netD
    end
end
