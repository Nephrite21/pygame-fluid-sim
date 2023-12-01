import pygame
import math
import numpy as np
from numba import njit
from numba import cuda

pygame.init()


#########################################################################
#SETTINGS
#########################################################################
#pygame settings
screenSize = [800,450]

#simulation settings
boundsSize = (600, 350)
gravity = 5
collisionDamping = 0.95
dt = 0
targetDensity = 1
pressureMultiplier = 1

#particle setting
    #particle align setting
particleNumber = 100
particleSize = 3
particleDistance = 10 # distance between particle's starting point
particleNumberInRow = 5
    #particle physics setting
particleMass = 1
smoothingRadius = 50.0

#DrawSetting
backgroundColor = (0,0,0)
boundBorderColor = (255,0,0)
particleColor = (255,255,255)

#gradient Setting
stepSize = 0.001
rightVec = np.array([1,0])
downVec = np.array([0,1])


#########################################################################
#CONFIGURATION
#########################################################################
#pygame configuration
screen = pygame.display.set_mode(screenSize)
running = True
clock = pygame.time.Clock()

#simulation configuration
boundsPosition = ((screenSize[0]-boundsSize[0])/2,(screenSize[1]-boundsSize[1])/2)
Center = (screenSize[0]/2,screenSize[1]/2)
halfboundSize = (boundsSize[0]/2 - particleSize, boundsSize[1]/2-particleSize)

#particle configuration
    #particle placing
if particleNumber > particleNumberInRow:
    particlePlacementWidth = (particleNumberInRow-1) * particleDistance
else:
    particlePlacementWidth = (particleNumber-1)* particleDistance
particlePlacementHeight = (particleNumber//particleNumberInRow)*particleDistance
particlePlacementStartingPoint = (Center[0] - particlePlacementWidth/2,Center[1] - particlePlacementHeight/2)
gg=[]
hh=[]
for i in range(0,particleNumber):
    gg.append([particlePlacementStartingPoint[0] + (i%particleNumberInRow)*particleDistance,
                     particlePlacementStartingPoint[1] + (i//particleNumberInRow)*particleDistance])
    hh.append([0.0,0.0])
position = np.array(gg)
velocity = np.array(hh)
particleProperties = np.zeros(position.shape[0])
particleDensities = np.zeros(position.shape[0])
#바보같이 써놓은 코드, 나중에 제대로 바꿀것

#particle smoothing variables
scale = 12/(3.141*pow(smoothingRadius,4))
volume = 3.141 * pow(smoothingRadius,4)/6


#########################################################################
#FUNCTIONS
#########################################################################
#functional functions
def magnitude(a,b=np.zeros(2)):
    squaredValue = 0
    for i in a-b:
        squaredValue += i*i
    return math.sqrt(squaredValue)

def normVec2(a):
    if a[0] == 0 and a[1] == 0:
        return np.zeros(2)
    else:
        size = math.sqrt(a[0]*a[0] + a[1]*a[1])
        b= [a[0]/size , a[1]/size]
        return np.array(b)


#Physics functions
def ResolveCollisions():
    for i in range(0,particleNumber):
        if abs(Center[0]-position[i][0]) > halfboundSize[0]:
            position[i][0] = math.copysign(halfboundSize[0], position[i][0]-Center[0])+Center[0]
            velocity[i][0] = velocity[i][0] * collisionDamping * -1
        if abs(Center[1]-position[i][1]) > halfboundSize[1]:
            position[i][1] = math.copysign(halfboundSize[1],position[i][1]-Center[1])+ Center[1]
            velocity[i][1] = velocity[i][1] * collisionDamping * -1

def ApplyGravity():
    velocity[:,1]+= gravity*dt
    #for positions in position:
    #   positions[1] = positions[1] + gravity*dt
    
def ApplyVelocity():
    global position
    position += velocity*dt
    
 

#fluid functions

def SmoothingKernel(dst):
    if(dst >= smoothingRadius):
        return 0.0
    return (smoothingRadius-dst) * (smoothingRadius-dst)/volume


def SmoothingKernelDerivates(dst):
    if dst >= smoothingRadius:
        return 0.0
    return (dst-smoothingRadius)*scale

#@cuda.jit
def CalculateDensity(samplePoint):
    a= position - samplePoint
    dstx2 = np.power(a[:,0],2) + np.power(a[:,1],2)
    dst = np.clip(smoothingRadius-np.sqrt(dstx2), a_min=0.0, a_max=None)
    influence = np.power(dst,2)/volume
    densities = particleMass*influence
    return np.sum(densities)

#@cuda.jit
#def gpu_CalculateDensity(position, samplePoint, smoothingRadius, volume, particleMass, density_out):
#    i = cuda.grid(1)
#    if i < position.shape[0]:
#        a = position[i] - samplePoint
#        dst = a[0] * a[0] + a[1] * a[1]
#        dst = max(dst - smoothingRadius, 0)
#        influence = dst / volume
#        density_out[i] = particleMass * influence

# CalculateDensity 함수를 GPU에서 실행하기 위해 래퍼 함수 정의
#def CalculateDensityGPU(position, samplePoint, smoothingRadius, volume, particleMass, density_out):
#    threads_per_block = 128
#    blocks_per_grid = (position.shape[0] + threads_per_block - 1) // threads_per_block
#    gpu_CalculateDensity[blocks_per_grid, threads_per_block](position, samplePoint, smoothingRadius, volume, particleMass, density_out)


def CalculateSharedPressure(densityA, densityB):
    pressureA = ConvertDensityToPressure(densityA)
    pressureB = ConvertDensityToPressure(densityB)
    return (pressureA+pressureB)/2


def CalculatePressureForce(particleIndex):
    pressureForce = np.array([0.0,0.0])
    for i in range(0,particleNumber):
        if i == particleIndex:
            continue
        offset = position[i] - position[particleIndex]
        dst = magnitude(offset)
        if dst == 0:
            dir = np.random.normal(2)
        else:
            dir = offset/dst
        slope = SmoothingKernelDerivates(dst)
        sharedPressure = CalculateSharedPressure(particleDensities[i], particleDensities[particleIndex])
        pressureForce += -sharedPressure * dir * slope * particleMass / particleDensities[i]
    return pressureForce


def UpdateDensities():
    for i in range(0,particleNumber):
        particleDensities[i] = CalculateDensity(position[i])


def ConvertDensityToPressure(density):
    return (density - targetDensity) * pressureMultiplier


def ApplyPressureToParticle():
    for i in range(0,particleNumber):
        pressureForce = CalculatePressureForce(i)
        pressureAcceleration = pressureForce / particleDensities[i]
        velocity[i] += pressureAcceleration * dt


def SimulatePhysics():
    UpdateDensities()
    #ApplyGravity()
    ApplyPressureToParticle()
    ApplyVelocity()
    ResolveCollisions()
   

#draw functions
def DrawBackground():
    screen.fill(backgroundColor)
    pygame.draw.rect(screen,boundBorderColor,[boundsPosition[0],boundsPosition[1],boundsSize[0],boundsSize[1]],1)

def DrawParticle():
    for i in range(0,particleNumber):
        pygame.draw.circle(screen, particleColor, (position[i][0],position[i][1]), particleSize)

def DrawAll():
    DrawBackground()
    DrawParticle()


########################################################################
#GAME ENGINE
########################################################################
while running:
    #Quit event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    dt = clock.tick(60)/100
    
    #simulate physics
    SimulatePhysics()
    
    #draw all
    DrawAll()
    
    #update screen
    pygame.display.flip()
pygame.quit()