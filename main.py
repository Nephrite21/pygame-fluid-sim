import pygame
import math
import numpy as np
from numba import njit,cuda

pygame.init()


#########################################################################
#SETTINGS
#########################################################################
#pygame settings
screenSize = [800,450]

#simulation settings
boundsSize = (200, 200)
gravity = 10
collisionDamping = 1
targetDensity = 1
pressureMultiplier = 4
deltaTimeSetting = "fixed" #clock and fixed. use fixed when computing speed is low
deltaTime = 0.16 #seconds per frame

#particle setting
    #particle align setting
particleNumber = 75
particleSize = 1
particleDistance = 5 # distance between particle's starting point
particleNumberInRow = 15
    #particle physics setting
particleMass = 1
smoothingRadius = 30

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
if deltaTimeSetting == "clock":
    dt = clock.tick(60)/100
elif deltaTimeSetting == "fixed":
    dt = deltaTime

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
particleProperties = np.zeros(particleNumber)
particleDensities = np.ones(particleNumber)
pressureForce = np.zeros((particleNumber,2))
spatialLookup = np.zeros((particleNumber,2))
startIndices = np.zeros(particleNumber)
smoothingRadiusX2 = smoothingRadius*smoothingRadius
cellOffsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
#바보같이 써놓은 코드, 나중에 제대로 바꿀것

#particle smoothing variables
scale = 12/(3.141*pow(smoothingRadius,4))
volume = 3.141 * pow(smoothingRadius,4)/6

#########################################################################
#Temporary
#########################################################################
time_to_Calc = []


#########################################################################
#FUNCTIONS
#########################################################################
#functional functions
def magnitude(a,b=np.zeros(2)):
    squaredValue = 0
    for i in a-b:
        squaredValue += i*i
    return math.sqrt(squaredValue)
def sqrMagnitude(a,b=np.zeros(2)):
    squaredValue = 0
    for i in a-b:
        squaredValue += i*i
    return squaredValue

def normVec2(a):
    if a[0] == 0 and a[1] == 0:
        return np.zeros(2)
    else:
        size = math.sqrt(a[0]*a[0] + a[1]*a[1])
        b= [a[0]/size , a[1]/size]
        return np.array(b)


#Physics functions
def ResolveCollisions():
    out_of_bounds_x = np.abs(Center[0] - position[:, 0]) > halfboundSize[0]
    out_of_bounds_y = np.abs(Center[1] - position[:, 1]) > halfboundSize[1]

    position[out_of_bounds_x, 0] = np.sign(position[out_of_bounds_x, 0] - Center[0]) * halfboundSize[0] + Center[0]
    velocity[out_of_bounds_x, 0] *= -collisionDamping

    position[out_of_bounds_y, 1] = np.sign(position[out_of_bounds_y, 1] - Center[1]) * halfboundSize[1] + Center[1]
    velocity[out_of_bounds_y, 1] *= -collisionDamping

def ApplyGravity():
    velocity[:,1]+= gravity*dt
    
def ApplyVelocity():
    global position
    position += velocity*dt
    
 

#fluid functions
def SmoothingKernel(dst):
    dst[dst >= smoothingRadius] = 0.0
    return np.maximum(smoothingRadius - dst, 0.0) * np.maximum(smoothingRadius - dst, 0.0) / volume


def SmoothingKernelDerivates(dst):
    if dst >= smoothingRadius:
        return 0.0
    return (dst-smoothingRadius)*scale

#@cuda.jit
def CalculateDensity(samplePoint):
    a= position - samplePoint
    dstX2 = np.power(a[:,0],2) + np.power(a[:,1],2) #distance squared
    dst = np.clip(smoothingRadius-np.sqrt(dstX2), a_min=0.0, a_max=None)
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
    return ((densityA - targetDensity) * pressureMultiplier+(densityB - targetDensity) * pressureMultiplier)/2


def CalculatePressureForce(particleIndex):
    pressureForce = np.zeros(2)
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
        if particleDensities[i] == 0:
            continue
        pressureForce += -sharedPressure * dir * slope * particleMass / particleDensities[i]
    return pressureForce


""" def CalculatePressureForce(particleIndex):
    pressureForce = np.array([0.0, 0.0])
    for i in range(particleNumber):
        if i == particleIndex:
            continue
        offset = position[i] - position[particleIndex]
        dst = np.linalg.norm(offset)
        dir = np.divide(offset, dst, out=np.zeros_like(offset), where=dst != 0)
        slope = SmoothingKernelDerivates(dst)
        sharedPressure = CalculateSharedPressure(particleDensities[i], particleDensities[particleIndex])
        pressureForce += -sharedPressure * dir * slope * particleMass / particleDensities[i]
    return pressureForce """

def UpdateDensities():
    global particleDensities
    particleDensities = np.array([CalculateDensity(pos) for pos in position])

def ApplyPressureToParticle():
    global pressureForce, velocity
    pressureForce = np.array([CalculatePressureForce(i) for i in range(particleNumber)])
    velocity += (pressureForce.T / particleDensities).T * dt


def SimulatePhysics():
    #UpdateSpatialLookup()
    UpdateDensities()
    #ForeachPointWithinRadius() ##not implemented yet
    
    ApplyPressureToParticle()
    ApplyGravity()
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

#optimization
def PositionToCellCoord(point):
    cellX = point[0]//smoothingRadius
    cellY = point[1]//smoothingRadius
    return (cellX,cellY)

def HashCell(cellX,cellY):
    return cellX * 15823 + cellY*9737333

def GetKeyFromHash(hash):
    return int(hash)%particleNumber

def UpdateSpatialLookup():
    global spatialLookup
    for i in range(0,particleNumber):
        cellX,cellY = PositionToCellCoord(position[i])
        cellKey = GetKeyFromHash(HashCell(cellX,cellY))
        spatialLookup[i] = (i,cellKey) #cellKey와 파티클 인덱스를 저장
        startIndices[i] = -1
    spatialLookup = np.sort(spatialLookup,axis=0) #cellKey를 기준으로 정렬
    for i in range(0,particleNumber):
        key = spatialLookup[i][1]
        if i == 0:
            keyPrev = -1
        else:
            keyPrev = spatialLookup[i-1][1]
        if key != keyPrev:
            startIndices[int(key)] = i

def ForeachPointWithinRadius(samplePoint):
    (centerX,centerY) = PositionToCellCoord(samplePoint,smoothingRadius)
    for (offsetX, offsetY) in cellOffsets:
        key = GetKeyFromHash(HashCell(centerX+offsetX,centerY+offsetY))
        cellStartIndex = startIndices[key]
        for i in range(cellStartIndex,spatialLookup.size()):
            if spatialLookup[i].cellKey != key:
                break
            particleIndex = spatialLookup[i][0]
            #sqrDst = sqrMagnitude(position[particleIndex], samplePoint)
            #반복해야 하는 함수 집어넣기

def ForeachPointWithinRadiusUpdateDensities(samplePoint):
    global particleDensities
    density = 0
    (centerX,centerY) = PositionToCellCoord(samplePoint,smoothingRadius) #위치를 기반으로 셀 좌표 구한다
    for (offsetX, offsetY) in cellOffsets: # 계산할 범위는 중앙 기준 1칸거리인 칸 안에서 해야할일은~
        key = GetKeyFromHash(HashCell(centerX+offsetX,centerY+offsetY)) #해쉬키를 구한다
        cellStartIndex = startIndices[key] # 해쉬키를 기반으로 시작 인덱스를 구한다.
        for i in range(cellStartIndex,spatialLookup.size()): #인덱스 끝까지
            if spatialLookup[i].cellKey != key:
                break
            particleIndex = spatialLookup[i][0]
            particlePosition = position[particleIndex]
            density += calcDens(samplePoint, particlePosition)

def calcDens(samplePoint,particlePosition):
    dst = magnitude(particlePosition,samplePoint)

            
#def UpdateDensities():
#    global particleDensities
#    particleDensities = np.array([CalculateDensity(pos) for pos in position])
#def CalculateDensity(samplePoint):
#    a= position - samplePoint
#    dstX2 = np.power(a[:,0],2) + np.power(a[:,1],2) #distance squared
#    # dst = np.clip(smoothingRadius-np.sqrt(dstX2), a_min=0.0, a_max=None)
#    dstX2[dstX2 > smoothingRadius] = 0
#    influence = dstX2/volume
#    #influence = np.power(dst,2)/volume
#    densities = particleMass*influence
#    return np.sum(densities)


########################################################################
#GAME ENGINE
########################################################################
while running:
    #Quit event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if deltaTimeSetting == "clock":
        dt = clock.tick(60)/100

    #simulate physics
    SimulatePhysics()
    
    #draw all
    DrawAll()
    
    #update screen
    pygame.display.flip()
pygame.quit()