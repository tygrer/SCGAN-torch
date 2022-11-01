-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua
-- code derived from https://github.com/soumith/dcgan.torch and https://github.com/phillipi/pix2pix

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
content = paths.dofile('util/content_loss.lua')
require 'image'
require 'models.architectures'

-- load configuration file
options = require 'options'
opt = options.parse_options('train')

-- setup visualization
visualizer = require 'util/visualizer'

-- initialize torch GPU/CPU mode
if opt.gpu > 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  print ("GPU Mode")
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
  print ("CPU Mode")
end

-- load data
local data_loader = nil
if opt.align_data > 0 then
  require 'data.aligned_data_loader'
  data_loader = AlignedDataLoader()
else
  require 'data.unaligned_data_loader'
  data_loader = UnalignedDataLoader()
  require 'data.aligned_data_loader'
   opt.DATA_ROOT = './datasets/align/'
  data_loader2 = AlignedDataLoader()
end
print( "DataLoader " .. data_loader:name() .. " was created.")
print( "DataLoader2 " .. data_loader2:name() .. " was created.")
opt.align_data=0;
opt.DATA_ROOT = './datasets/train1000/'
data_loader:Initialize(opt)
opt.align_data=1;
opt.DATA_ROOT = './datasets/align/'
data_loader2:Initialize(opt)
opt.DATA_ROOT = './datasets/train1000/'
opt.align_data=0;
-- set batch/instance normalization
set_normalization(opt.norm)
print( opt.model .. " is ")
--- timer
local epoch_tm = torch.Timer()
local tm = torch.Timer()

-- define model
local model = nil
local display_plot = nil
if opt.model == 'cycle_gan' then
  assert(data_loader:name() == 'UnalignedDataLoader')
  require 'models.cycle_gan_model'
  model = CycleGANModel()
elseif opt.model == 'pix2pix' then
  require 'models.pix2pix_model'
  assert(data_loader:name() == 'AlignedDataLoader')
  model = Pix2PixModel()
elseif opt.model == 'bigan' then
  assert(data_loader:name() == 'UnalignedDataLoader')
  require 'models.bigan_model'
  model = BiGANModel()
elseif opt.model == 'content_gan' then
  require 'models.content_gan_model'
  assert(data_loader:name() == 'UnalignedDataLoader')
  model = ContentGANModel()
else
  error('Please specify a correct model')
end

-- print the model name
print('Model ' .. model:model_name() .. ' was specified.')
model:Initialize(opt)

-- set up the loss plot
require 'util/plot_util'
plotUtil = PlotUtil()
display_plot = model:DisplayPlot(opt)
plotUtil:Initialize(display_plot, opt.display_id, opt.name)

--------------------------------------------------------------------------------
-- Helper Functions
--------------------------------------------------------------------------------
function visualize_current_results()
  local visuals = model:GetCurrentVisuals(opt)
  for i,visual in ipairs(visuals) do
    visualizer.disp_image(visual.img, opt.display_winsize,
                          opt.display_id+i, opt.name .. ' ' .. visual.label)
  end
end

function save_current_results(epoch, counter)
  local visuals = model:GetCurrentVisuals(opt)
  visualizer.save_results(visuals, opt, epoch, counter)
end

function print_current_errors(epoch, counter_in_epoch)
  print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
           .. '%s'):
      format(epoch, ((counter_in_epoch-1) / opt.batchSize),
      math.floor(math.min(data_loader:size(), opt.ntrain) / opt.batchSize),
      tm:time().real / opt.batchSize,
      data_loader:time_elapsed_to_fetch_data() / opt.batchSize,
      model:GetCurrentErrorDescription()
  ))
end

function plot_current_errors(epoch, counter_ratio, opt)
  local errs = model:GetCurrentErrors(opt)
  local plot_vals = { epoch + counter_ratio}
  plotUtil:Display(plot_vals, errs)
end

--------------------------------------------------------------------------------
-- Main Training Loop
--------------------------------------------------------------------------------
local counter = 1
local num_batches = math.floor(math.min(data_loader:size()+data_loader2:size(), opt.ntrain) / opt.batchSize)
print('#training iterations: ' .. opt.niter+opt.niter_decay )

for epoch = 1, opt.niter+opt.niter_decay do
    epoch_tm:reset()
    --for counter_in_epoch = 1, math.min(data_loader:size()+2*data_loader2:size(), opt.ntrain), opt.batchSize do
	for counter_in_epoch = 1, math.min(data_loader:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        opt.counter = counter
		
		--if counter%2==0 then
			
			--local real_data2A, real_data2B, _, _= data_loader2:GetNextBatchtrain()
			--model:ForwardL1({real_A2=real_data2A, real_B2=real_data2B}, opt)
			-- run backward pass
			--print('Ddddd')
            --model:OptimizeParametersL1G(opt)
			--model:OptimizeParametersL1D(opt)
			--if counter%2==0 then
				
			--end					
		--else
			local real_dataA, real_dataB, _, _,data_CSRA, data_CSRB= data_loader:GetNextBatchtrain()
			model:ForwardCYCGAN({real_A=real_dataA, real_B=real_dataB, CSRA=data_CSRA, CSRB=data_CSRB}, opt)
			-- run backward pass
			--print('Gggg')
            model:OptimizeParametersCYCGANG(opt)
		    model:OptimizeParametersCYCGAND(opt)
			--if counter % opt.display_freq == 0 and opt.display_id > 0 then
			--visualize_current_results()
			--end

        -- logging
			if counter % opt.print_freq == 0 then
			print_current_errors(epoch, counter_in_epoch)
			plot_current_errors(epoch, counter_in_epoch/num_batches, opt)
			end

        -- save latest model
			if counter % opt.save_latest_freq == 0 and counter > 0 then
			print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
			model:Save('latest', opt)
			end

        -- save latest results
			if counter % opt.save_display_freq == 0 then
			save_current_results(epoch, counter)
			end
		    
		--end
        -- display on the web server
        counter = counter + 1
    end

    -- save model at the end of epoch
    if epoch % opt.save_epoch_freq == 0 then
        print(('saving the model (epoch %d, iters %d)'):format(epoch, counter))
        model:Save('latest', opt)
        model:Save(epoch, opt)
   end
    -- print the timing information after each epoch
    print(('End of epoch %d / %d \t Time Taken: %.3f'):
        format(epoch, opt.niter+opt.niter_decay, epoch_tm:time().real))

    -- update learning rate
    if epoch > opt.niter then
      model:UpdateLearningRate(opt)
    end
    -- refresh parameters
    model:RefreshParameters(opt)
end
