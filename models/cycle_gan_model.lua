local class = require 'class'
require 'models.base_model'
require 'models.architectures'
require 'util.image_pool'
require 'loadcaffe'
require 'libcuda_utils'
require 'cutorch'
require 'cunn'
require 'image'
local matio = require 'matio'
util = paths.dofile('../util/util.lua')
CycleGANModel = class('CycleGANModel', 'BaseModel')

function CycleGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function CycleGANModel:model_name()
  return 'CycleGANModel'
end

function CycleGANModel:InitializeStates(use_wgan)
  optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end
-- Defines models and networks
function CycleGANModel:Initialize(opt)
  if opt.test == 0 then
    self.fakeAPool = ImagePool(opt.pool_size)
    self.fakeBPool = ImagePool(opt.pool_size)
    self.fakeA2Pool = ImagePool(opt.pool_size)
    self.fakeB2Pool = ImagePool(opt.pool_size)
  end
  -- define tensors
  self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.real_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.real_A2 = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.real_B2 = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.fake_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.fake_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.fake_A2 = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.fake_B2 = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  self.rec_A  = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.rec_B  = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
  -- load/define models
  local use_lsgan = ((opt.use_lsgan ~= nil) and (opt.use_lsgan == 1))
  if not use_lsgan then
    self.criterionGAN = nn.BCECriterion()
  else
    self.criterionGAN = nn.MSECriterion()
  end
  self.criterionRec = nn.AbsCriterion()
  self.criterionL1 = nn.AbsCriterion()
  local netG_A, netD_A, netG_B, netD_B = nil, nil, nil, nil
  if opt.continue_train == 1 then
    if opt.which_epoch then -- which_epoch option exists in test mode
      netG_A = util.load_test_model('G_A', opt)
      netG_B = util.load_test_model('G_B', opt)
      netD_A = util.load_test_model('D_A', opt)
      netD_B = util.load_test_model('D_B', opt)
    else
      netG_A = util.load_model('G_A', opt)
      netG_B = util.load_model('G_B', opt)
      netD_A = util.load_model('D_A', opt)
      netD_B = util.load_model('D_B', opt)
    end
	
  else
    local use_sigmoid = (not use_lsgan)
    -- netG_test = defineG(opt.input_nc, opt.output_nc, opt.ngf, "resnet_unet", opt.arch)
    -- os.exit()
    netG_A = defineG(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,opt.growthRate,opt.bottleneck,opt.dropRate,opt.arch)
    
    netD_A = defineD(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, use_sigmoid)  -- no sigmoid layer
    print('netD_A...', netD_A)
    netG_B = defineG(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, opt.growthRate,opt.bottleneck,opt.dropRate,opt.arch)
    print('netG_B...', netG_B)
    netD_B = defineD(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, use_sigmoid)  -- no sigmoid layer
    print('netD_B', netD_B)
  end


  self.netD_A = netD_A
  self.netG_A = netG_A
  self.netG_B = netG_B
  self.netD_B = netD_B

  -- define real/fake labels
  local D_A_size = self.netD_A:forward(self.real_B):size() 
      -- hack: assume D_size_A = D_size_B
  print('D_A_size:',D_A_size)
  self.fake_label_A = torch.Tensor(D_A_size):fill(0.0)
  self.real_label_A = torch.Tensor(D_A_size):fill(0.9) -- no soft smoothing
  local D_B_size = self.netD_B:forward(self.real_A):size()  -- hack: assume D_size_A = D_size_B
  print('D_B_size:',D_B_size)
  self.fake_label_B = torch.Tensor(D_B_size):fill(0.0)
  self.real_label_B = torch.Tensor(D_B_size):fill(0.9) -- no soft smoothing
  --local D_B2_size = self.netD_B:forward(self.real_A2):size()  print('D_B2_size:',D_B2_size)
  self.fake_label_A2=torch.Tensor(D_A_size):fill(0.0)
  self.real_label_A2=torch.Tensor(D_A_size):fill(0.9) 
  self.fake_label_B2=torch.Tensor(D_B_size):fill(0.0)
  self.real_label_B2=torch.Tensor(D_B_size):fill(0.9) 
  -- local D_A_size2 = self.netD_A:forward(self.real_B2):size()  -- hack: assume D_size_A = D_size_B
  -- self.fake_label_A2 = torch.Tensor(D_A_size2):fill(0.0)
  -- self.real_label_A2 = torch.Tensor(D_A_size2):fill(0.9) -- no soft smoothing
  -- local D_B_size2 = self.netD_B:forward(self.real_A2):size()  -- hack: assume D_size_A = D_size_B
  --self.fake_label_B2 = torch.Tensor(D_B_size2):fill(0.0)
  -- self.real_label_B2 = torch.Tensor(D_B_size2):fill(0.9) -- no soft smoothing

  self.optimStateD_A = self:InitializeStates()
  self.optimStateG_A = self:InitializeStates()
  self.optimStateD_B = self:InitializeStates()
  self.optimStateG_B = self:InitializeStates()

  self:RefreshParameters()
  --self.A_idx = {{}, {1, opt.input_nc}, {}, {}}
  --self.B_idx = {{}, {opt.input_nc+1, opt.input_nc+opt.output_nc}, {}, {}}
  print('---------- # Learnable Parameters --------------')
  print(('G_A = %d'):format(self.parametersG_A:size(1)))
  print(('D_A = %d'):format(self.parametersD_A:size(1)))
  print(('G_B = %d'):format(self.parametersG_B:size(1)))
  print(('D_B = %d'):format(self.parametersD_B:size(1)))
  print('------------------------------------------------')
  -- os.exit()
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function CycleGANModel:ForwardCYCGAN(input, opt)
  if opt.which_direction == 'BtoA' then
  	local temp = input.real_A:clone()
  	input.real_A = input.real_B:clone()
  	input.real_B = temp
	--temp1=input.real_A2:clone()
	--input.real_A2=input.real_B2
	--input.real_B2=temp1
	local temp2 = input.CSRA:clone()
  	input.CSRA = input.CSRB:clone()
  	input.CSRB = temp2

	--real_A=real_dataA, real_B=real_dataB,real_A2=real_data2A, real_B2=real_data2B, CSRA=data_CSRA, CSRB=data_CSRB,CSRA2=data_CSR2A, CSRB2=data_CSR2B
  end

  --self.real_A2:copy(input.real_A2)
  --self.real_B2:copy(input.real_B2)
  self.real_A:copy(input.real_A)
   
  --print(input.real_A)
  self.real_B:copy(input.real_B)
  --self.real_A2:copy(input.real_A2)
  --print(input.real_A2)
  --self.real_B2:copy(input.real_B2)
  --self.fake_AB2[self.A_idx]:copy(self.real_A2)
  --self.fake_B= self.netG_A:forward(self.real_A):clone()
  --self.fake_A = self.netG_B:forward(self.real_B):clone()
  --self.rec_A  = self.netG_B:forward(self.fake_B):clone()
  --self.rec_B  = self.netG_A:forward(self.fake_A):clone()
  --self.fake_AB2[self.B_idx]:copy(self.fake_B2)
--  print(type(input.CSRA))
  self.CSRA=input.CSRA
--  print(type(self.CSRA))
  self.CSRB=input.CSRB

  if opt.test == 1 then  -- forward for test
    -- print('test time: generate images')
    self.fake_B = self.netG_A:forward(self.real_A):clone()
    self.fake_A = self.netG_B:forward(self.real_B):clone()
	--self.fake_B2 = self.netG_A:forward(self.real_A2):clone()
    --self.fake_A2 = self.netG_B:forward(self.real_B2):clone()
    self.rec_A  = self.netG_B:forward(self.fake_B):clone()
    self.rec_B  = self.netG_A:forward(self.fake_A):clone()
	--self.rec_A2  = self.netG_B:forward(self.fake_B2):clone()
    --self.rec_B2  = self.netG_A:forward(self.fake_A2):clone()
  end
end
function CycleGANModel:ForwardL1(input, opt)
  if opt.which_direction == 'BtoA' then
  	--local temp = input.real_A:clone()
  	--input.real_A = input.real_B:clone()
  	--input.real_B = temp
	temp1=input.real_A2:clone()
	input.real_A2=input.real_B2
	input.real_B2=temp1
	--local temp2 = input.CSRA:clone()
  	--input.CSRA = input.CSRB:clone()
  	--input.CSRB = temp2

	--real_A=real_dataA, real_B=real_dataB,real_A2=real_data2A, real_B2=real_data2B, CSRA=data_CSRA, CSRB=data_CSRB,CSRA2=data_CSR2A, CSRB2=data_CSR2B
  end

  self.real_A2:copy(input.real_A2)
  self.real_B2:copy(input.real_B2)
  --self.real_A:copy(input.real_A)
   
  --print(input.real_A)
  --self.real_B:copy(input.real_B)
  --self.real_A2:copy(input.real_A2)
  --print(input.real_A2)
  --self.real_B2:copy(input.real_B2)
  --self.fake_AB2[self.A_idx]:copy(self.real_A2)
  --self.fake_B2 = self.netG_A:forward(self.real_A2):clone()
  --self.fake_A2 = self.netG_B:forward(self.real_B2):clone()
  --self.fake_AB2[self.B_idx]:copy(self.fake_B2)
--  print(type(input.CSRA))
  --self.CSRA=input.CSRA
--  print(type(self.CSRA))
  --self.CSRB=input.CSRB

  if opt.test == 1 then  -- forward for test
    -- print('test time: generate images')
    self.fake_B = self.netG_A:forward(self.real_A):clone()
    self.fake_A = self.netG_B:forward(self.real_B):clone()
	--self.fake_B2 = self.netG_A:forward(self.real_A2):clone()
    --self.fake_A2 = self.netG_B:forward(self.real_B2):clone()
    self.rec_A  = self.netG_B:forward(self.fake_B):clone()
    self.rec_B  = self.netG_A:forward(self.fake_A):clone()
	--self.rec_A2  = self.netG_B:forward(self.fake_B2):clone()
    --self.rec_B2  = self.netG_A:forward(self.fake_A2):clone()
  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
function CycleGANModel:fDx_basicCYCGAN(x, gradParams, netD, netG, real, fake, real_label, fake_label, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()
  -- Real  log(D_A(B))
  local output1 = netD:forward(real)
  --print('output1',output1)
  local errD_real = self.criterionGAN:forward(output1, real_label)
  local df_do = self.criterionGAN:backward(output1, real_label)
  netD:backward(real, df_do)
  
  -- local output2 = netD:forward(real2)
  --print('output2',output2)
  -- local errD_real2 = self.criterionGAN:forward(output2, real_label2)
  -- local df_do2 = self.criterionGAN:backward(output2, real_label2)
  -- netD:backward(real2, df_do2)
  -- Fake  + log(1 - D_A(G_A(A)))
  local output3 = netD:forward(fake)

  local errD_fake = self.criterionGAN:forward(output3, fake_label)
  local df_do3 = self.criterionGAN:backward(output3, fake_label)
  netD:backward(fake, df_do3)

  -- local output4 = netD:forward(fake2)

  -- local errD_fake2 = self.criterionGAN:forward(output4, fake_label2)

  -- local df_do22 = self.criterionGAN:backward(output4, fake_label2)

  -- netD:backward(fake2, df_do22)
  -- Compute loss
  local errD = (errD_real + errD_fake ) / 2.0
  --print('errD',errD)
  return errD, gradParams
end

function CycleGANModel:fDx_basicL1(x, gradParams, netD, netG, real2, fake2, real_label2, fake_label2, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()
  -- Real  log(D_A(B))
  -- local output1 = netD:forward(real)
  --print('output1',output1)
  -- local errD_real = self.criterionGAN:forward(output1, real_label)
  -- local df_do = self.criterionGAN:backward(output1, real_label)
  -- netD:backward(real, df_do)
  
  local output2 = netD:forward(real2)
  --print('output2',output2)
  local errD_real2 = self.criterionGAN:forward(output2, real_label2)
  local df_do2 = self.criterionGAN:backward(output2, real_label2)
  netD:backward(real2, df_do2)
  -- Fake  + log(1 - D_A(G_A(A)))
  -- local output3 = netD:forward(fake)
  --print('output3',output3)
  -- local errD_fake = self.criterionGAN:forward(output3, fake_label)
  -- local df_do3 = self.criterionGAN:backward(output3, fake_label)
  -- netD:backward(fake, df_do3)

  local output4 = netD:forward(fake2)
  --print('fake',fake)
  local errD_fake2 = self.criterionGAN:forward(output4, fake_label2)

  --print('fake_label2',fake_label2)
  --print('errD_fake2',errD_fake2)
  local df_do22 = self.criterionGAN:backward(output4, fake_label2)

  netD:backward(fake2, df_do22)
  -- Compute loss
  local errD = ( errD_real2 + errD_fake2) / 2.0
  --print('errD',errD)
  return errD, gradParams
end

function CycleGANModel:fDAxL1(x, opt)
  -- use image pool that stores the old fake images
  --fake_B = self.fakeBPool:Query(self.fake_B)
  --print('self.fake_B',self.fake_B)
  --self.fake_B2 = self.netG_B:forward(self.real2_A):clone()
  fake_B2 = self.fakeB2Pool:Query(self.fake_B2)
  --print('self.fake_B2',self.fake_B2)
  self.errD_A, gradParams = self:fDx_basicL1(x, self.gradparametersD_A, self.netD_A, self.netG_A,
                             self.real_B2, fake_B2, self.real_label_A2, self.fake_label_A2, opt)
  return self.errD_A, gradParams
end
function CycleGANModel:fDAxCYCGAN(x, opt)
  -- use image pool that stores the old fake images
  fake_B = self.fakeBPool:Query(self.fake_B)
  --print('self.fake_B',self.fake_B)
  --self.fake_B2 = self.netG_B:forward(self.real2_A):clone()
  --fake_B2 = self.fakeB2Pool:Query(self.fake_B2)
  --print('self.fake_B2',self.fake_B2)
  self.errD_A, gradParams = self:fDx_basicCYCGAN(x, self.gradparametersD_A, self.netD_A, self.netG_A,
                            self.real_B, fake_B, self.real_label_A, self.fake_label_A, opt)
  return self.errD_A, gradParams
end

function CycleGANModel:fDBxCYCGAN(x, opt)
  -- use image pool that stores the old fake images
  fake_A = self.fakeAPool:Query(self.fake_A)
  --fake_A2 = self.fakeA2Pool:Query(self.fake_A2)
  self.errD_B, gradParams = self:fDx_basicCYCGAN(x, self.gradparametersD_B, self.netD_B, self.netG_B,
                            self.real_A, fake_A, self.real_label_B, self.fake_label_B, opt)
  return self.errD_B, gradParams
end
function CycleGANModel:fDBxL1(x, opt)
  -- use image pool that stores the old fake images
  --fake_A = self.fakeAPool:Query(self.fake_A)
  fake_A2 = self.fakeA2Pool:Query(self.fake_A2)
  self.errD_B, gradParams = self:fDx_basicL1(x, self.gradparametersD_B, self.netD_B, self.netG_B,
                            self.real_A2, fake_A2, self.real_label_B2, self.fake_label_B2, opt)
  return self.errD_B, gradParams
end


function CycleGANModel:fGx_basicCYCGAN(x, gradParams, netG, netD, netE, real, real2,real_label, CSR, lambda1, lambda2, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  util.BiasZero(netE)  -- inverse mapping
  gradParams:zero()

  -- G should be identity if real2 is fed.
  local errI = nil
  local identity = nil
  -- if opt.identity > 0 then
    -- identity = netG:forward(real2):clone()
    -- errI = self.criterionRec:forward(identity, real2)*lambda2*opt.identity
    -- local didentity_loss_do = self.criterionRec:backward(identity, real2):mul(lambda2):mul(opt.identity)
    -- netG:backward(real2, didentity_loss_do)
  -- end
 -- First. G(A) should fake the discriminator
  --itorch.image(real_alignA)
  --local fake_alignB = netG:forward(real_alignA):clone()
  --itorch.image(fake_alignB)
  --local output_alignB = netD:forward(fake_alignB)
  --print(type(output_alignB))
  -- local errG2 = self.criterionGAN:forward(output_alignB, real_label2)
  -- local dgan_loss_dd2 = self.criterionGAN:backward(output_alignB, real_label2)
  -- local dgan_loss_do = netD:updateGradInput(fake_alignB, dgan_loss_dd2)

  -- Second. G(A) should be close to the real
  -- local errL1 = self.criterionL1:forward(fake_alignB, real_alignB) * 10
  -- local dl1_loss_do = self.criterionL1:backward(fake_alignB, real_alignB) * 10
  -- netG:backward(real_alignA, (dgan_loss_do + dl1_loss_do)/2)
  
  --- GAN loss: D_A(G_A(A))
  local fake = netG:forward(real):clone()
  local output = netD:forward(fake)
  local errG = self.criterionGAN:forward(output, real_label)
  local df_do1 = self.criterionGAN:backward(output, real_label)
  local df_d_GAN = netD:updateGradInput(fake, df_do1)
  
  ------------------------------------
  -- forward cycle loss
  local rec = netE:forward(fake):clone()
  local errRec = self.criterionRec:forward(rec,real) * lambda1
  local df_do2 = self.criterionRec:backward(rec, real):mul(lambda1)
  local df_do_rec = netE:updateGradInput(fake, df_do2)

  local c,h,w =rec:size(1),rec:size(2),rec:size(3)
    -- load matting laplacian
  local gradient_LocalAffine = MattingLaplacian(rec, CSR, w, w):mul(2)
  --print(errRec)
    -- if num_calls % params.save_iter == 0 then
      -- local best = SmoothLocalAffine(output, input, 1e-7, 3, h, w, 7, 0.05)
      -- fn = 'serial_example' .. '/best' .. tostring(index) .. '_t_' .. tostring(num_calls) .. '.png'
      -- image.save(fn, best)
    -- end 
  netG:backward(real, df_d_GAN + df_do_rec+gradient_LocalAffine)
  --add-------------------------

  -- backward cycle loss
  local fake2 = netE:forward(real2)--:clone()
  local rec2 = netG:forward(fake2)--:clone()
  local errAdapt = self.criterionRec:forward(rec2, real2) * lambda2
  local df_do_coadapt = self.criterionRec:backward(rec2, real2):mul(lambda2)
  netG:backward(fake2, df_do_coadapt)
	-- local grad = torch.add(gradient_VggNetwork, gradient_LocalAffine)
  return gradParams, errG, errRec, errI, errL1,fake,fake_alignB, rec, identity
end
function CycleGANModel:fGx_basicL1(x, gradParams, netG, netD, real_alignA, real_alignB, real_label2, lambda1, lambda2, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  --util.BiasZero(netE)  -- inverse mapping
  gradParams:zero()

  -- G should be identity if real2 is fed.
  local errI = nil
  local identity = nil
  -- if opt.identity > 0 then
    -- identity = netG:forward(real2):clone()
    -- errI = self.criterionRec:forward(identity, real2)*lambda2*opt.identity
    -- local didentity_loss_do = self.criterionRec:backward(identity, real2):mul(lambda2):mul(opt.identity)
    -- netG:backward(real2, didentity_loss_do)
  -- end
 -- First. G(A) should fake the discriminator
  --itorch.image(real_alignA)
  local fake_alignB = netG:forward(real_alignA):clone()
  --itorch.image(fake_alignB)
  local output_alignB = netD:forward(fake_alignB)
  --print(type(output_alignB))
  local errG2 = self.criterionGAN:forward(output_alignB, real_label2)
  local dgan_loss_dd2 = self.criterionGAN:backward(output_alignB, real_label2)
  local dgan_loss_do = netD:updateGradInput(fake_alignB, dgan_loss_dd2)

  -- Second. G(A) should be close to the real
  local errL1 = self.criterionL1:forward(fake_alignB, real_alignB) * 10
  local dl1_loss_do = self.criterionL1:backward(fake_alignB, real_alignB) * 10
  netG:backward(real_alignA, (dgan_loss_do + dl1_loss_do)/2)
  
  --- GAN loss: D_A(G_A(A))
  -- local fake = netG:forward(real):clone()
  -- local output = netD:forward(fake)
  -- local errG = self.criterionGAN:forward(output, real_label)
  -- local df_do1 = self.criterionGAN:backward(output, real_label)
  -- local df_d_GAN = netD:updateGradInput(fake, df_do1)
  ----------------------------------
  --forward cycle loss
  -- local rec = netE:forward(fake):clone()
  -- local errRec = self.criterionRec:forward(rec,real) * lambda1
  -- local df_do2 = self.criterionRec:backward(rec, real):mul(lambda1)
  -- local df_do_rec = netE:updateGradInput(fake, df_do2)
  -- local c,h,w =rec:size(1),rec:size(2),rec:size(3)
    --load matting laplacian
  -- local gradient_LocalAffine = MattingLaplacian(rec, CSR, h, w):mul(0.01)
    --if num_calls % params.save_iter == 0 then
      --local best = SmoothLocalAffine(output, input, 1e-7, 3, h, w, 7, 0.05)
      --fn = 'serial_example' .. '/best' .. tostring(index) .. '_t_' .. tostring(num_calls) .. '.png'
      --image.save(fn, best)
    --end 
  -- netG:backward(real, df_d_GAN + df_do_rec+gradient_LocalAffine)
  --add-------------------------

  -- backward cycle loss
  -- local fake2 = netE:forward(real2)--:clone()
  -- local rec2 = netG:forward(fake2)--:clone()
  -- local errAdapt = self.criterionRec:forward(rec2, real2) * lambda2
  -- local df_do_coadapt = self.criterionRec:backward(rec2, real2):mul(lambda2)
  -- netG:backward(fake2, df_do_coadapt)
	-- local grad = torch.add(gradient_VggNetwork, gradient_LocalAffine)
  return gradParams, errG2, errRec, errI, errL1,fake,fake_alignB, rec, identity
end

function MattingLaplacian(output, CSR, h, w)
  local N, c = CSR:size(1), CSR:size(2)
  local CSR_rowIdx = torch.CudaIntTensor(N):copy(torch.round(CSR[{{1,-1},1}]))
  local CSR_colIdx = torch.CudaIntTensor(N):copy(torch.round(CSR[{{1,-1},2}]))
  local CSR_val    = torch.CudaTensor(N):copy(CSR[{{1,-1},3}])

  local output01 = torch.div(output, 256.0)

  local grad = cuda_utils.matting_laplacian(output01, h, w, CSR_rowIdx, CSR_colIdx, CSR_val, N)
  
  grad:div(256.0)
  return grad
end
function CycleGANModel:fGAxL1(x, opt)
  self.gradparametersG_A, self.errG_A, self.errRec_A, self.errI_A, self.errL1_A, self.fake_B,self.fake_alignB, self.rec_A, self.identity_B =
  self:fGx_basicL1(x, self.gradparametersG_A, self.netG_A, self.netD_A, self.real_A2, self.real_B2,self.real_label_A2, opt.lambda_A, opt.lambda_B, opt)
  return self.errG_A, self.gradparametersG_A
end
function CycleGANModel:fGAxCYCGAN(x, opt)
  self.gradparametersG_A, self.errG_A, self.errRec_A, self.errI_A, self.errL1_A, self.fake_B,self.fake_alignB, self.rec_A, self.identity_B =
  self:fGx_basicCYCGAN(x, self.gradparametersG_A, self.netG_A, self.netD_A, self.netG_B, self.real_A, self.real_B, self.real_label_A, self.CSRB, opt.lambda_A, opt.lambda_B, opt)
  return self.errG_A, self.gradparametersG_A
end
function CycleGANModel:fGBxL1(x, opt)
  self.gradparametersG_B, self.errG_B, self.errRec_B, self.errI_B, self.errL1_B,self.fake_A,self.fake_alignA, self.rec_B, self.identity_A =
  self:fGx_basicL1(x, self.gradparametersG_B, self.netG_B, self.netD_B,self.real_B2,self.real_A2, self.real_label_B2, opt.lambda_B, opt.lambda_A, opt)
  return self.errG_B, self.gradparametersG_B
end
function CycleGANModel:fGBxCYCGAN(x, opt)
  self.gradparametersG_B, self.errG_B, self.errRec_B, self.errI_B, self.errL1_B,self.fake_A,self.fake_alignA, self.rec_B, self.identity_A =
  self:fGx_basicCYCGAN(x, self.gradparametersG_B, self.netG_B, self.netD_B, self.netG_A, self.real_B, self.real_A, self.real_label_B, self.CSRA, opt.lambda_B, opt.lambda_A, opt)
  return self.errG_B, self.gradparametersG_B
end

function CycleGANModel:OptimizeParametersCYCGANG(opt)
  --local fDA = function(x) return self:fDAxCYCGAN(x, opt) end
  local fGA = function(x) return self:fGAxCYCGAN(x, opt) end
  --local fDB = function(x) return self:fDBxCYCGAN(x, opt) end
  -- if opt.forward_cycle == 1 then
  local fGB = function(x) return self:fGBxCYCGAN(x, opt) end
  -- end
  optim.adam(fGA, self.parametersG_A, self.optimStateG_A)
  --optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
  optim.adam(fGB, self.parametersG_B, self.optimStateG_B)
  --optim.adam(fDB, self.parametersD_B, self.optimStateD_B)
end
function CycleGANModel:OptimizeParametersCYCGAND(opt)
  --local fDA = function(x) return self:fDAxCYCGAN(x, opt) end
  local fDA = function(x) return self:fDAxCYCGAN(x, opt) end
  --local fDB = function(x) return self:fDBxCYCGAN(x, opt) end
  -- if opt.forward_cycle == 1 then
  local fDB = function(x) return self:fDBxCYCGAN(x, opt) end
  -- end
  optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
  --optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
  optim.adam(fDB, self.parametersD_B, self.optimStateD_B)
  --optim.adam(fDB, self.parametersD_B, self.optimStateD_B)
end

function CycleGANModel:OptimizeParametersL1G(opt)
  --local fDA = function(x) return self:fDAxL1(x, opt) end
  local fGA = function(x) return self:fGAxL1(x, opt) end
  --local fDB = function(x) return self:fDBxL1(x, opt) end
  -- if opt.forward_cycle == 1 then
  local fGB = function(x) return self:fGBxL1(x, opt) end
  -- end
  optim.adam(fGA, self.parametersG_A, self.optimStateG_A)
  --optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
  optim.adam(fGB, self.parametersG_B, self.optimStateG_B)
  --optim.adam(fDB, self.parametersD_B, self.optimStateD_B)
end
function CycleGANModel:OptimizeParametersL1D(opt)
  local fDA = function(x) return self:fDAxL1(x, opt) end
  --local fGA = function(x) return self:fGAxL1(x, opt) end
  local fDB = function(x) return self:fDBxL1(x, opt) end
  -- if opt.forward_cycle == 1 then
  --local fGB = function(x) return self:fGBxL1(x, opt) end
  -- end
  --optim.adam(fGA, self.parametersG_A, self.optimStateG_A)
  optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
  --optim.adam(fGB, self.parametersG_B, self.optimStateG_B)
  optim.adam(fDB, self.parametersD_B, self.optimStateD_B)
end

function CycleGANModel:RefreshParameters()
  self.parametersD_A, self.gradparametersD_A = nil, nil -- nil them to avoid spiking memory
  self.parametersG_A, self.gradparametersG_A = nil, nil
  self.parametersG_B, self.gradparametersG_B = nil, nil
  self.parametersD_B, self.gradparametersD_B = nil, nil
  -- define parameters of optimization
  self.parametersG_A, self.gradparametersG_A = self.netG_A:getParameters()
  self.parametersD_A, self.gradparametersD_A = self.netD_A:getParameters()
  self.parametersG_B, self.gradparametersG_B = self.netG_B:getParameters()
  self.parametersD_B, self.gradparametersD_B = self.netD_B:getParameters()
end

function CycleGANModel:Save(prefix, opt)
  util.save_model(self.netG_A, prefix .. '_net_G_A.t7', 1)
  util.save_model(self.netD_A, prefix .. '_net_D_A.t7', 1)
  util.save_model(self.netG_B, prefix .. '_net_G_B.t7', 1)
  util.save_model(self.netD_B, prefix .. '_net_D_B.t7', 1)
end

function CycleGANModel:GetCurrentErrorDescription()
  description = ('[A] G: %.4f  D: %.4f  L1：%.4f Rec: %.4f I: %.4f || [B] G: %.4f D: %.4f L1：%.4f Rec: %.4f I:%.4f'):format(
                         self.errG_A and self.errG_A or -1,
                         self.errD_A and self.errD_A or -1,
						 self.errL1_A and self.errL1_A or -1,
                         self.errRec_A and self.errRec_A or -1,
                         self.errI_A and self.errI_A or -1,
                         self.errG_B and self.errG_B or -1,
                         self.errD_B and self.errD_B or -1,
						 self.errL1_B and self.errL1_B or -1,
                         self.errRec_B and self.errRec_B or -1,
                         self.errI_B and self.errI_B or -1)
  return description
end

function CycleGANModel:GetCurrentErrors()
  local errors = {errG_A=self.errG_A, errD_A=self.errD_A, errL1_A=self.errL1_A,errRec_A=self.errRec_A, errI_A=self.errI_A,errG_B=self.errG_B, errD_B=self.errD_B, errL1_B=self.errL1_B,errRec_B=self.errRec_B, errI_B=self.errI_B}
  return errors
end

-- returns a string that describes the display plot configuration
function CycleGANModel:DisplayPlot(opt)
  if opt.identity > 0 then
    return 'errG_A,errD_A,errRec_A,errI_A,errG_B,errD_B,errRec_B,errI_B'
  else
    return 'errG_A,errD_A,errRec_A,errG_B,errD_B,errRec_B'
  end
end

function CycleGANModel:UpdateLearningRate(opt)
  local lrd = opt.lr / opt.niter_decay
  local old_lr = self.optimStateD_A['learningRate']
  local lr =  old_lr - lrd
  self.optimStateD_A['learningRate'] = lr
  self.optimStateD_B['learningRate'] = lr
  self.optimStateG_A['learningRate'] = lr
  self.optimStateG_B['learningRate'] = lr
  print(('update learning rate: %f -> %f'):format(old_lr, lr))
end

local function MakeIm3(im)
  if im:size(2) == 1 then
    local im3 = torch.repeatTensor(im, 1,3,1,1)
    return im3
  else
    return im
  end
end

function CycleGANModel:GetCurrentVisuals(opt, size)
  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.real_A), label='real_A'})
  table.insert(visuals, {img=MakeIm3(self.fake_B), label='fake_B'})
  table.insert(visuals, {img=MakeIm3(self.rec_A), label='rec_A'})
  if opt.test == 0 and opt.identity > 0 then
    table.insert(visuals, {img=MakeIm3(self.identity_A), label='identity_A'})
  end
  table.insert(visuals, {img=MakeIm3(self.real_B), label='real_B'})
  table.insert(visuals, {img=MakeIm3(self.fake_A), label='fake_A'})
  table.insert(visuals, {img=MakeIm3(self.rec_B), label='rec_B'})
  if opt.test == 0 and opt.identity > 0 then
    table.insert(visuals, {img=MakeIm3(self.identity_B), label='identity_B'})
  end
  return visuals
end
