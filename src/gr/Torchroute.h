#pragma once

class TorchPin
{
public:
	int x;
	int y;
	int layer;
	void set(int x, int y, int layer);
};

class TorchEdge
{
public:
	TorchPin pin1;
	TorchPin pin2;
	void set(TorchPin pin1, TorchPin pin2);
};