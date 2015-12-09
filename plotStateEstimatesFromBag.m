function [ ] = plotStateEstimatesFromBag( filename, varargin )
%plotStateEstimatesFromBag plots state estimates from ROS bag file
%   Digests file described by arg filename.
%   
%   plotStateEstimatesFromBag(filename, startOffset, endOffset) will only
%   plot a subset of the data generated between startOffset and endOffset
%   seconds into the bag file. These arguments are recommended because otherwise the
%   amount of data can be overwhelming. 
%
%   Currently-handled estimates:
%     Boondoggler
%     Vicon

close all;

% add path to custom message
addpath('../matlab_gen/msggen');

startOffset = 0;
endOffset = -1;
if nargin > 1
  startOffset = varargin{1};
end
if nargin > 2
  endOffset = varargin{2};
end

%% Define data sources
sources = cell(3,1);

sources{1}.name = 'Estimator';
sources{1}.topic_pose = '/boondoggler/pose';
sources{1}.topic_vel = '/boondoggler/vel';
sources{1}.rotate_vel = false;

sources{2}.name = 'Vicon';
sources{2}.topic_pose = '/Batman/pose';
sources{2}.topic_vel = '/Batman/vel';
sources{2}.rotate_vel = true;

sources{3}.name = 'Vicon -- f450';
sources{3}.topic_pose = '/f450/pose';
sources{3}.topic_vel = '/f450/vel';
sources{3}.rotate_vel = true;

%% Populate data from bag
S = size(sources,1);
disp(sprintf('Parsing bag file at %s', filename));
bag = rosbag(filename);

time_window = [bag.StartTime + startOffset, bag.StartTime + endOffset];
if endOffset == -1
  time_window(2) = bag.EndTime;
end

window_duration = time_window(2) - time_window(1);
if window_duration > 150 && nargin < 2
  disp(sprintf('Warning: data window is %f seconds, may be too much data.', window_duration));
  disp('Consider providing optional startOffset and endOffset arguments.');
end

for s = 1:S
  disp(sprintf('Extracting %s pose data...', sources{s}.name));
  pose_msgs = select(bag, 'Time', time_window, 'Topic', sources{s}.topic_pose );
  
  t_series_pose = timeseries(pose_msgs, 'Pose.Position.X', 'Pose.Position.Y', 'Pose.Position.Z', 'Pose.Orientation.W', 'Pose.Orientation.X', 'Pose.Orientation.Y', 'Pose.Orientation.Z');
  
  if isempty(t_series_pose)
    disp(sprintf('No %s data found on topic %s.', sources{s}.name, sources{s}.topic_pose));
    sources{s}.empty = true;
    continue
  else
    sources{s}.empty = false;
  end
  
  % now extract initial state to normalize position
  first_pose_msgs = readMessages(select(bag,'Time',[bag.StartTime,bag.StartTime+1], 'Topic', sources{s}.topic_pose));
  first_pos = [first_pose_msgs{1}.Pose.Position.X, first_pose_msgs{1}.Pose.Position.Y, first_pose_msgs{1}.Pose.Position.Z];
  
  sources{s}.ts = bsxfun(@minus, t_series_pose.Time, bag.StartTime);
  sources{s}.pos = bsxfun(@minus, t_series_pose.Data(:,1:3), first_pos);
  sources{s}.q = t_series_pose.Data(:,4:7);
  
  % convert quaternions to euler angles
  [roll,pitch,yaw] = quat_to_euler(sources{s}.q);
  sources{s}.euler = [roll,pitch,yaw];
  
  disp(sprintf('Extracting %s vel data...', sources{s}.name));
  vel_msgs = select(bag, 'Time', time_window, 'Topic', sources{s}.topic_vel );
  
  t_series_vel = timeseries(vel_msgs, 'Twist.Linear.X', 'Twist.Linear.Y', 'Twist.Linear.Z');
  sources{s}.vel_ts = bsxfun(@minus, t_series_vel.Time, bag.StartTime);
  sources{s}.vel = t_series_vel.Data(:,1:3);
  if sources{s}.rotate_vel == true
    disp('rotating...')
    % assume we don't need to time-align quaternions to velocity data
    n = min(size(sources{s}.q,1), size(sources{s}.vel,1));
    sources{s}.vel = quatrotate(sources{s}.q(1:n,:), sources{s}.vel(1:n,:));
  end
  
end

clear first_pose_msgs;
clear pose_msgs;
clear t_series;

% %% Now pull estimator state
disp('Extracting estimator state data...');
state_msgs = select(bag, 'Time', time_window, 'Topic', '/boondoggler/status' );
t_series_state = timeseries(state_msgs, 'GyroBiases.X', 'GyroBiases.Y', 'GyroBiases.Z', 'DragCoefficient', 'SpThrust', 'FlightState', 'DisturbMode');
status.ts = bsxfun(@minus, t_series_state.Time, bag.StartTime);
status.gyro_biases = t_series_state.Data(:,1:3);
status.drag_coeff = t_series_state.Data(:,4);
status.sp_thrust = t_series_state.Data(:,5);
status.flight_state = t_series_state.Data(:,6);
status.disturb_mode = t_series_state.Data(:,7);

%% Determine flight windows
flight_start_ts = status.ts(find( diff(status.flight_state > 0) == 1 ) + 1);
flight_end_ts = status.ts(find( diff(status.flight_state > 0) == -1 ));
if numel(flight_end_ts) < numel(flight_start_ts)
  flight_end_ts = [flight_end_ts; status.ts(end)];
end
flight_ts = [flight_start_ts, flight_end_ts];

ground_start_ts = status.ts(find( diff(status.flight_state == 0) == 1 ) + 1);
ground_end_ts = status.ts(find( diff(status.flight_state == 0) == -1 ));
ground_start_ts = [status.ts(1); ground_start_ts];
if numel(ground_end_ts) < numel(ground_start_ts)
  ground_end_ts = [ground_end_ts; status.ts(end)];
end
ground_ts = [ground_start_ts, ground_end_ts];

draw_flight_start = @(i) line([flight_start_ts(i),flight_start_ts(i)], [-100,100], 'Color', [0,0,0]);

%% Plot data
disp('Plotting');

figure;
title('Altitude');
xlabel('Time (s)');

figure;
title('XY Position');
ylabel('Y');
xlabel('X');

figure;
subplot(2,2,1);
title('qw');
subplot(2,2,2);
title('qx');
subplot(2,2,3);
title('qy');
subplot(2,2,4);
title('qz');

figure;
subplot(2,1,1);
title('roll');
subplot(2,1,2);
title('pitch');

figure;
title('yaw');

figure;
subplot(3,1,1);
title('vel_u');
subplot(3,1,2);
title('vel_v');
subplot(3,1,3);
title('vel_w');

for s = 1:S
  if sources{s}.empty
    continue
  end
  
  figure(1);
  hold all;
  plot(sources{s}.ts, sources{s}.pos(:,3));

  figure(2);
  hold all;
  plot(sources{s}.pos(:,1), sources{s}.pos(:,2));
  
  figure(3);
  subplot(2,2,1);
  hold all;
  plot(sources{s}.ts, sources{s}.q(:,1));
  subplot(2,2,2);
  hold all;
  plot(sources{s}.ts, sources{s}.q(:,2));
  subplot(2,2,3);
  hold all;
  plot(sources{s}.ts, sources{s}.q(:,3));
  subplot(2,2,4);
  hold all;
  plot(sources{s}.ts, sources{s}.q(:,4));
  
  figure(4);
  subplot(2,1,1);
  hold all;
  plot(sources{s}.ts, rad2deg(sources{s}.euler(:,1)));
  axis manual;
  draw_flight_start(1);
  subplot(2,1,2);
  hold all;
  plot(sources{s}.ts, rad2deg(sources{s}.euler(:,2)));
  axis manual;
  draw_flight_start(1);
  
  figure(5);
  hold all;
  polar(sources{s}.euler(:,3), sources{s}.ts);
  
  figure(6);
  subplot(3,1,1);
  hold all;
  plot(sources{s}.vel_ts, sources{s}.vel(:,1));
  axis manual;
  draw_flight_start(1);
  subplot(3,1,2);
  hold all;
  plot(sources{s}.vel_ts, sources{s}.vel(:,2));
  axis manual;
  draw_flight_start(1);
  subplot(3,1,3);
  hold all;
  plot(sources{s}.vel_ts, sources{s}.vel(:,3));
  axis manual;
  draw_flight_start(1);
  
  
end

names = {};
for s = 1:S
  if ~sources{s}.empty
    names = {names{:}, sources{s}.name};
  end
end
figure(1);
axis manual;
draw_flight_start(1);
legend(names);
figure(2);
legend(names);
figure(3);
legend(names);
figure(4);
legend(names);
figure(5);
legend(names);
figure(6);
legend(names);

%% Plot state variables
figure;
hold all;
plot(status.ts, status.gyro_biases(:,1));
plot(status.ts, status.gyro_biases(:,2));
plot(status.ts, status.gyro_biases(:,3));
title('Gyro Biases');
xlabel('Time (s)');
legend('X', 'Y', 'Z');
axis manual;
draw_flight_start(1);

figure;
plot(status.ts, status.drag_coeff);
title('Specific Drag Coefficient');
xlabel('Time (s)');
axis manual;
draw_flight_start(1);

figure;
plot(status.ts, status.sp_thrust);
title('Specific Thrust');
xlabel('Time (s)');
axis manual;
draw_flight_start(1);

end

