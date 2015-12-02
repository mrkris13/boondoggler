function [ roll, pitch, yaw ] = quat_to_euler( quat )
%  Converts from quaternion to Body 321 Euler representation
%   quat is Nx4 matrix (x,y,z,w)
%   roll is Nx1 matrix
%   pitch is Nx1 matrix
%   yaw is Nx1 matrix

yaw = atan2(2* (quat(:,1).*quat(:,4) + quat(:,2).*quat(:,3) ), 1 - 2*(quat(:,1).^2 + quat(:,2).^2));
pitch = asin(2 * (quat(:,2).*quat(:,4) - quat(:,1).*quat(:,3)) );
roll = atan2(2 * (quat(:,3).*quat(:,4) + quat(:,1).*quat(:,2)), 1 - 2*(quat(:,2).^2 + quat(:,3).^2));

end

